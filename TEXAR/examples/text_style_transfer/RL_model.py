# only convnet

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier,Conv1DNetwork,BidirectionalRNNEncoder
from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size
        
        
class RLModel(object):
    # inputs: mixed gender data
    # finputs: female data
    # minputs: male data
    def __init__(self, inputs, finputs, minputs, vocab,gamma,hparams=None):
        self._hparams = tx.HParams(hparams, None)
        self._build_model(inputs, vocab, finputs, minputs,gamma)

    def _build_model(self, inputs, vocab, finputs, minputs,gamma):
        """Builds the model.
        """
        self.inputs = inputs
        self.finputs = finputs
        self.minputs = minputs
        self.vocab = vocab
        
        self.embedder = WordEmbedder(
            vocab_size=self.vocab.size,
            hparams=self._hparams.embedder)
        # maybe later have to try BidirectionalLSTMEncoder
        self.encoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder) #GRU cell

        # text_ids for encoder, with BOS(begin of sentence) token removed
        self.enc_text_ids = self.inputs['text_ids'][:, 1:]
        self.enc_outputs, self.final_state = self.encoder(self.embedder(self.enc_text_ids),sequence_length=self.inputs['length']-1)

        h = self.final_state

        # Teacher-force decoding and the auto-encoding loss for G
        self.decoder = AttentionRNNDecoder(
            memory=self.enc_outputs,
            memory_sequence_length=self.inputs['length']-1,
            cell_input_fn=lambda inputs, attention: inputs, 
            #default: lambda inputs, attention: tf.concat([inputs, attention], -1), which cancats regular RNN cell inputs with attentions.
            vocab_size=self.vocab.size,
            hparams=self._hparams.decoder)

        self.connector = MLPTransformConnector(self.decoder.state_size)

        self.g_outputs, _, _ = self.decoder(
            initial_state=self.connector(h), inputs=self.inputs['text_ids'],
            embedding=self.embedder, sequence_length=self.inputs['length']-1) 

        self.loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=self.inputs['text_ids'][:, 1:],
            logits=self.g_outputs.logits,
            sequence_length=self.inputs['length']-1,
            average_across_timesteps=True,
            sum_over_timesteps=False)              
        
        # Greedy decoding, used in eval (and RL training)
        start_tokens = tf.ones_like(self.inputs['labels']) * self.vocab.bos_token_id
        end_token = self.vocab.eos_token_id
        self.outputs, _, length = self.decoder(
            #也许可以尝试之后把这个换成 "infer_sample"看效果
            decoding_strategy='infer_greedy', initial_state=self.connector(h),
            embedding=self.embedder, start_tokens=start_tokens, end_token=end_token)
        
        
        # Creates optimizers
        self.g_vars = collect_trainable_variables([self.embedder, self.encoder, self.connector, self.decoder])       
        self.train_op_g_ae = get_train_op(self.loss_g_ae, self.g_vars, hparams=self._hparams.opt)
        
        # Interface tensors
        self.samples = {
            "batch_size": get_batch_size(self.inputs['text_ids']),
            "original": self.inputs['text_ids'][:, 1:],
            "transferred": self.outputs.sample_id #outputs 是infer_greedy的结果
        }
        
        
        ############################ female sentiment regression model
        #现在只用了convnet不知道效果，之后可以试试RNN decoding看regression的准确度，或者把两个结合一下（concat成一个向量）
        self.fconvnet = Conv1DNetwork(hparams=self._hparams.convnet) #[batch_size, time_steps, embedding_dim] (default input)
        #convnet = Conv1DNetwork()
        self.freg_embedder = WordEmbedder(vocab_size=self.vocab.size,hparams=self._hparams.embedder) #(64, 26, 100) (output shape of clas_embedder(ids=inputs['text_ids'][:, 1:]))
        self.fconv_output = self.fconvnet(inputs=self.freg_embedder(ids=self.finputs['text_ids'][:, 1:])) #(64, 128)  等一会做一下finputs!!!
        p = {"type": "Dense", "kwargs": {'units':1}}
        self.fdense_layer = tx.core.layers.get_layer(hparams=p)
        self.freg_output = self.fdense_layer(inputs=self.fconv_output) 
        
        '''
        #考虑
        self.fenc_text_ids = self.finputs['text_ids'][:, 1:]
        self.fencoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder) #GRU cell
        self.fenc_outputs, self.ffinal_state = self.fencoder(self.freg_embedder(self.fenc_text_ids),sequence_length=self.finputs['length']-1)
        self.freg_output = self.fdense_layer(inputs = tf.concat([self.fconv_output, self.ffinal_state], -1))
        '''
        
        self.fprediction = tf.reshape(self.freg_output,[-1])
        self.fground_truth = tf.to_float(self.finputs['labels'])

        self.floss_reg_single = tf.pow(self.fprediction - self.fground_truth,2) #这样得到的是单个的loss,可以之后在RL里面对一整个batch进行update
        self.floss_reg_batch = tf.reduce_mean(self.floss_reg_single) #对一个batch求和平均的loss

        #self.freg_vars = collect_trainable_variables([self.freg_embedder, self.fconvnet, self.fencoder, self.fdense_layer])
        self.freg_vars = collect_trainable_variables([self.freg_embedder, self.fconvnet, self.fdense_layer])
        self.ftrain_op_d = get_train_op(self.floss_reg_batch, self.freg_vars, hparams=self._hparams.opt)
  
        self.freg_sample = {
            "fprediction": self.fprediction,
            "fground_truth": self.fground_truth,
            "fsent": self.finputs['text_ids'][:, 1:]
        }
              
            
            
        ############################ male sentiment regression model
        self.mconvnet = Conv1DNetwork(hparams=self._hparams.convnet) #[batch_size, time_steps, embedding_dim] (default input)
        #convnet = Conv1DNetwork()
        self.mreg_embedder = WordEmbedder(vocab_size=self.vocab.size,hparams=self._hparams.embedder) #(64, 26, 100) (output shape of clas_embedder(ids=inputs['text_ids'][:, 1:]))
        self.mconv_output = self.mconvnet(inputs=self.mreg_embedder(ids=self.minputs['text_ids'][:, 1:])) #(64, 128)
        p = {"type": "Dense", "kwargs": {'units':1}}
        self.mdense_layer = tx.core.layers.get_layer(hparams=p)
        self.mreg_output = self.mdense_layer(inputs=self.mconv_output)
        
        '''
        #考虑
        self.menc_text_ids = self.minputs['text_ids'][:, 1:]
        self.mencoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder) #GRU cell
        self.menc_outputs, self.mfinal_state = self.mencoder(self.mreg_embedder(self.menc_text_ids),sequence_length=self.minputs['length']-1)
        self.mreg_output = self.mdense_layer(inputs = tf.concat([self.mconv_output, self.mfinal_state], -1))
        '''

        self.mprediction = tf.reshape(self.mreg_output,[-1])
        self.mground_truth = tf.to_float(self.minputs['labels'])

        self.mloss_reg_single = tf.pow(self.mprediction - self.mground_truth,2) #这样得到的是单个的loss,可以之后在RL里面对一整个batch进行update
        self.mloss_reg_batch = tf.reduce_mean(self.mloss_reg_single) #对一个batch求和平均的loss

        #self.mreg_vars = collect_trainable_variables([self.mreg_embedder, self.mconvnet, self.mencoder, self.mdense_layer])
        self.mreg_vars = collect_trainable_variables([self.mreg_embedder, self.mconvnet,self.mdense_layer])
        self.mtrain_op_d = get_train_op(self.mloss_reg_batch, self.mreg_vars, hparams=self._hparams.opt)
 
        self.mreg_sample = {
                "mprediction": self.mprediction,
                "mground_truth": self.mground_truth,
                "msent": self.minputs['text_ids'][:, 1:]
        }
        
        ###### get self.pre_dif when doing RL training (for transferred sents)
        ### pass to female regression model
        self.RL_fconv_output = self.fconvnet(inputs=self.freg_embedder(ids=self.outputs.sample_id)) #(64, 128)  等一会做一下finputs!!!
        self.RL_freg_output = self.fdense_layer(inputs=self.RL_fconv_output)
        self.RL_fprediction = tf.reshape(self.RL_freg_output,[-1])
        ### pass to male regression model
        self.RL_mconv_output = self.mconvnet(inputs=self.mreg_embedder(ids=self.outputs.sample_id)) #(64, 128)  等一会做一下finputs!!!
        self.RL_mreg_output = self.mdense_layer(inputs=self.RL_mconv_output)
        self.RL_mprediction = tf.reshape(self.RL_mreg_output,[-1])
        
        self.pre_dif = tf.abs(self.RL_fprediction-self.RL_mprediction)
        
        
        ###### get self.Ypre_dif for original sents
        ### pass to female regression model
        self.YRL_fconv_output = self.fconvnet(inputs=self.freg_embedder(ids=self.inputs['text_ids'][:, 1:])) #(64, 128)  等一会做一下finputs!!!
        self.YRL_freg_output = self.fdense_layer(inputs=self.YRL_fconv_output)
        self.YRL_fprediction = tf.reshape(self.YRL_freg_output,[-1])
        ### pass to male regression model
        self.YRL_mconv_output = self.mconvnet(inputs=self.mreg_embedder(ids=self.inputs['text_ids'][:, 1:])) #(64, 128)  等一会做一下finputs!!!
        self.YRL_mreg_output = self.mdense_layer(inputs=self.YRL_mconv_output)
        self.YRL_mprediction = tf.reshape(self.YRL_mreg_output,[-1])

        self.Ypre_dif = tf.abs(self.YRL_fprediction-self.YRL_mprediction)
              
        
        
        ######################## RL training
        '''
        def fil(elem):
            return tf.where(elem > 1.3, tf.minimum(elem,3), 0)
        def fil_pushsmall(elem):
            return tf.add(tf.where(elem <0.5, 1, 0),tf.where(elem>1.5,-0.5*elem,0))
        '''
        
        '''
        #缩小prediction差异
        def fil1(elem):
            return tf.where(elem<0.5,1.0,0.0)
        def fil2(elem):
            return tf.where(elem>1.5,-0.5*elem,0.0)
        '''
        
        #扩大prediction差异
        def fil1(elem):
            return tf.where(elem<0.5,-0.01,0.0)
        def fil2(elem):
            return tf.where(elem>1.3,elem,0.0)
        
        # 维数是(batch_size,time_step),对应的是一个batch中每一个sample的每一个timestep的loss
        self.beginning_loss_g_RL2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
            _sentinel=None,
            labels=self.outputs.sample_id,
            logits=self.outputs.logits,
            name=None
        )
        self.middle_loss_g_RL2 = tf.reduce_sum(self.beginning_loss_g_RL2,axis=1) #(batch_size,),这样得到的loss是每一个句子的loss(对time_steps求和，对batch不求和)
       
        #trivial "RL" training with all weight set to 1
        #final_loss_g_RL2 = tf.reduce_sum(self.middle_loss_g_RL2)
        
        #RL training
        self.filtered = tf.add(tf.map_fn(fil1,self.pre_dif),tf.map_fn(fil2,self.pre_dif))
        self.updated_loss_per_sent = tf.multiply(self.filtered,self.middle_loss_g_RL2) #haven't set threshold for weight update
        self.updated_loss_per_batch = tf.reduce_sum(self.updated_loss_per_sent) #############！！有一个问题：
        # 我想update每一个句子的loss,但是train_updated那里会报错，所以好像只能updateloss的求和，这样是相当于update每一个句子的loss吗？
        
        self.vars_updated = collect_trainable_variables([self.connector, self.decoder])       
        self.train_updated = get_train_op(self.updated_loss_per_batch, self.vars_updated, hparams=self._hparams.opt)
        self.train_updated_interface = {
            "pre_dif": self.pre_dif,
            "updated_loss_per_sent":self.updated_loss_per_sent,
            "updated_loss_per_batch": self.updated_loss_per_batch,
        }       
        
        
        ### Train AE and RL together
        self.loss_AERL = gamma*self.updated_loss_per_batch+self.loss_g_ae
        self.vars_AERL = collect_trainable_variables([self.connector, self.decoder])       
        self.train_AERL = get_train_op(self.loss_AERL, self.vars_AERL, hparams=self._hparams.opt)