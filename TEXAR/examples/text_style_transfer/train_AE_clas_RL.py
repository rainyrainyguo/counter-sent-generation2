from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import importlib
import numpy as np
import tensorflow as tf
import texar as tx

import tensorflow as tf

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier,Conv1DNetwork,BidirectionalRNNEncoder
from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size

from RL_model import RLModel

tf.set_random_seed(1)



config = importlib.import_module('RLconfig')

# Data
train_data = tx.data.MultiAlignedData(config.train_data)
val_data = tx.data.MultiAlignedData(config.val_data)
test_data = tx.data.MultiAlignedData(config.test_data)
vocab = train_data.vocab(0)

iterator = tx.data.FeedableDataIterator({'train_g': train_data,'val': val_data, 'test': test_data})
batch = iterator.get_next()

# female Data
ftrain_data = tx.data.MultiAlignedData(config.ftrain_data)
fval_data = tx.data.MultiAlignedData(config.fval_data)
ftest_data = tx.data.MultiAlignedData(config.ftest_data)

fiterator = tx.data.FeedableDataIterator({'ftrain_g': ftrain_data,'fval': fval_data, 'ftest': ftest_data})
fbatch = fiterator.get_next()

# male Data
mtrain_data = tx.data.MultiAlignedData(config.mtrain_data)
mval_data = tx.data.MultiAlignedData(config.mval_data)
mtest_data = tx.data.MultiAlignedData(config.mtest_data)

miterator = tx.data.FeedableDataIterator({'mtrain_g': mtrain_data,'mval': mval_data, 'mtest': mtest_data})
mbatch = miterator.get_next()

gamma = 0.05
model = RLModel(batch, fbatch, mbatch, vocab, 0.05, config.model)






#################### define functions

def _train_epoch_AE(sess, epoch, verbose=True):
    avg_meters_g = tx.utils.AverageRecorder(size=5)

    step = 0
    while True:
        try:
            step += 1
            
            feed_dict = {iterator.handle: iterator.get_handle(sess, 'train_g')}
            vals_g = sess.run(model.train_op_g_ae, feed_dict=feed_dict)
            avg_meters_g.add(vals_g)

            if verbose and (step == 1 or step % 5 == 0):
                print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))

            '''
            if verbose and step % 2 == 0:
                iterator.restart_dataset(sess, 'val')
                _eval_epoch(sess, epoch)
            '''

        except tf.errors.OutOfRangeError:
            print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
            break
            
def _eval_epoch_AE(sess, epoch, val_or_test='val'):
    avg_meters = tx.utils.AverageRecorder()
    while True:
        try:
            feed_dict = {
                iterator.handle: iterator.get_handle(sess, val_or_test),
                tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
            }

            vals = sess.run(model.samples, feed_dict=feed_dict)
            batch_size = vals.pop('batch_size')

            # Computes BLEU
            samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
            hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
            print("samples: ",hyps)

            refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
            refs = np.expand_dims(refs, axis=1)
            print("reference: ",refs)

            bleu = tx.evals.corpus_bleu_moses(refs, hyps)
            vals['bleu'] = bleu

            avg_meters.add(vals, weight=batch_size)

            ###################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Writes samples
            '''
            tx.utils.write_paired_text(
                refs.squeeze(), hyps,
                os.path.join(config.sample_path, 'val.%d'%epoch),
                append=True, mode='v')
            '''
        except tf.errors.OutOfRangeError:
            print('{}: {}'.format(
                val_or_test, avg_meters.to_str(precision=4)))
            break
    return avg_meters.avg()

def f_train_epoch_reg(sess, epoch):
    avg_meters_g = tx.utils.AverageRecorder(size=5)
    step = 0
    while True:
        try:
            step += 1          
            feed_dict = {fiterator.handle: fiterator.get_handle(sess, 'ftrain_g')}
            vals_g = sess.run(model.ftrain_op_d, feed_dict=feed_dict)
            avg_meters_g.add(vals_g)

            if (step == 1 or step % 5 == 0):
                print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
            '''
            if step % 2 == 0:
                iterator.restart_dataset(sess, 'val')
                _eval_epoch(sess, epoch)
            '''
        except tf.errors.OutOfRangeError:
            print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
            break
            
def m_train_epoch_reg(sess, epoch):
    avg_meters_g = tx.utils.AverageRecorder(size=5)
    step = 0
    while True:
        try:
            step += 1           
            feed_dict = {miterator.handle: miterator.get_handle(sess, 'mtrain_g')}
            vals_g = sess.run(model.mtrain_op_d, feed_dict=feed_dict)
            avg_meters_g.add(vals_g)

            if (step == 1 or step % 5 == 0):
                print('step: {}, {}'.format(step, avg_meters_g.to_str(4)))
            '''
            if verbose and step % 2 == 0:
                iterator.restart_dataset(sess, 'val')
                _eval_epoch(sess, epoch)
            '''
        except tf.errors.OutOfRangeError:
            print('epoch: {}, {}'.format(epoch, avg_meters_g.to_str(4)))
            break

def fevalf_epoch_reg(sess, val_or_test='fval'):
    avg_meters = []
    while True:
        try:
            ffeed_dict = {
                fiterator.handle: fiterator.get_handle(sess, val_or_test),
                tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
            }
            vals = sess.run(model.floss_reg_batch, feed_dict=ffeed_dict)
            #batch_size = config.train_data['batch_size']

            avg_meters.append(vals)

        except tf.errors.OutOfRangeError:
            avg_loss = np.mean(avg_meters)
            print('{}: {}'.format(
                val_or_test, avg_loss))
            break
    return avg_loss

def mevalm_epoch_reg(sess, val_or_test='mval'):
    avg_meters = []
    while True:
        try:
            mfeed_dict = {
                miterator.handle: miterator.get_handle(sess, val_or_test),
                tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
            }
            vals = sess.run(model.mloss_reg_batch, feed_dict=mfeed_dict)
            #batch_size = config.train_data['batch_size']

            avg_meters.append(vals)

        except tf.errors.OutOfRangeError:
            avg_loss = np.mean(avg_meters)
            print('{}: {}'.format(
                val_or_test, avg_loss))
            break
    return avg_loss


def RL_train_epoch(sess,epoch):
    step = 0
    while True:
        try:
            step+=1
            feed_dict = {iterator.handle: iterator.get_handle(sess,'train_g')}
            vals_g = sess.run(model.train_updated_interface,feed_dict=feed_dict)
            
            if step==1 or step%5==0:
                print('step: {}, {}'.format(step,vals_g))
                
        except tf.errors.OutOfRangeError:
            print('epoch: {}, {}'.format(epoch, vals_g))
            break


def get_sents_AE(sess,val_or_test='test'): 
    iterator.initialize_dataset(sess)
    sample_sents=[]
    #ref_sents=[]
    pre_dif=[]
    i=1
    while True:
        print("batch: ",i)
        i=i+1
        try:
            feed_dict = {
                iterator.handle: iterator.get_handle(sess, val_or_test),
                tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
            }

            vals = sess.run({'a':model.samples,'b':model.pre_dif}, feed_dict=feed_dict)

            #batch_size = vals['a'].pop('batch_size')

            # Computes BLEU
            samples = tx.utils.dict_pop(vals['a'], list(model.samples.keys()))
            hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)
            #print("samples: ",hyps)

            #refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
            #refs = np.expand_dims(refs, axis=1)
            #print("reference: ",refs)
            
            sample_sents.extend(hyps.tolist())
            pre_dif.extend(vals['b'].tolist())
            #ref_sents.extend(refs.tolist())
            
            #dif = np.abs(predict_sentiment(str(hyps[0]),frnn,fTEXT)-predict_sentiment(str(hyps[0]),mrnn,mTEXT))

        except tf.errors.OutOfRangeError:
            print("all batches finished")
            break
    return sample_sents,pre_dif



############# start session

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
sess.run(tf.tables_initializer())

saver = tf.train.Saver(max_to_keep=None)

iterator.initialize_dataset(sess)
# fiterator 和 miterator 其实只是在训练discriminator的时候用，其他时候是用iterator
fiterator.initialize_dataset(sess)
miterator.initialize_dataset(sess)


################# train female and male sentiment classifier
print('Train female regression model')
for epoch in range(1, 20):
    fiterator.restart_dataset(sess, ['ftrain_g'])
    f_train_epoch_reg(sess,epoch)
    if epoch%3==0:
        print("evaluation on fval:")
        fiterator.restart_dataset(sess)
        fevalf_epoch_reg(sess, val_or_test='fval')

print('Train male regression model')
for epoch in range(1, 20):
    miterator.restart_dataset(sess, ['mtrain_g'])
    m_train_epoch_reg(sess,epoch)
    if epoch%3==0:
        print("evaluation on mval:")
        miterator.restart_dataset(sess)
        mevalm_epoch_reg(sess, val_or_test='mval')

print('saving model:RLsave/AE_clas100onlyconvnet_RL.ckpt')
saver.save(sess,'RLsave/AE_clas100onlyconvnet_RL.ckpt') # model after RL training (narrowing pre_dif while keeping the original sent unchanged)

fiterator.restart_dataset(sess)
print('fval evaluation loss: ',fevalf_epoch_reg(sess, val_or_test='fval'))
miterator.restart_dataset(sess)
print('mval evaluation loss: ',mevalf_epoch_reg(sess, val_or_test='mval'))

      