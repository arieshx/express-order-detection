# -*- coding:utf-8 -*-
from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from lib.roi_data_layer.layer import RoIDataLayer
from lib.utils.timer import Timer
from lib.roi_data_layer import roidb as rdl_roidb
from lib.fast_rcnn.config import cfg

_DEBUG = False

class SolverWrapper(object):
    def __init__(self, sess, network, imdb, roidb, imdb_test, roidb_test, output_dir, logdir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.imdb_test = imdb_test
        self.roidb_test = roidb_test
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print('Computing bounding-box regression targets...')
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print('done')

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100,write_version=tf.train.SaverDef.V2)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                             graph=tf.get_default_graph(),
                                             flush_secs=5)

    def snapshot(self, sess, iter):
        net = self.net
        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers and cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

        if cfg.TRAIN.BBOX_REG and 'bbox_pred' in net.layers:
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))

    def build_image_summary(self):
        # A simple graph for write image summary

        log_image_data = tf.placeholder(tf.uint8, [None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        # import tensorflow.python.ops.gen_logging_ops as logging_ops
        from tensorflow.python.ops import gen_logging_ops
        from tensorflow.python.framework import ops as _ops
        log_image = gen_logging_ops._image_summary(log_image_name, tf.expand_dims(log_image_data, 0), max_images=1)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        # log_image = tf.summary.image(log_image_name, tf.expand_dims(log_image_data, 0), max_outputs=1)
        return log_image, log_image_data, log_image_name


    def get_test_loss(self, sess, training_flag, data_layer_test, total_loss,model_loss, rpn_cross_entropy, side_loss,  v_loss, o_loss):
        """
        calculate loss in test set
        :param data_layer_test:
        :param total_loss:
        :param model_loss:
        :param rpn_cross_entropy:
        :param rpn_loss_box:
        :return:
        """
        from lib.roi_data_layer.minibatch import get_minibatch
        import random

        img_num = data_layer_test._perm.size
        total_loss_test,model_loss_test, rpn_loss_cls_test, side_loss_test, v_loss_test, o_loss_test = 0,0,0,0, 0, 0
        sample_num = 20
        list_sample = random.sample(list(range(img_num)), sample_num)
        for i in list_sample:
            minibatch_db = [data_layer_test._roidb[i]]
            _ = get_minibatch(minibatch_db, data_layer_test._num_classes)
            blobs = _

            # print(i,'  blobs:', blobs['im_info'])
            feed_dict = {
                self.net.data: blobs['data'],
                self.net.img_name: blobs['im_name'],
                self.net.im_info: blobs['im_info'],
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: blobs['gt_boxes'],
                self.net.gt_ishard: blobs['gt_ishard'],
                self.net.dontcare_areas: blobs['dontcare_areas'],
                training_flag: True
            }

            a, b, c, d, e, f = sess.run(fetches=[total_loss, model_loss, rpn_cross_entropy, side_loss, v_loss ,o_loss],
                       feed_dict=feed_dict)
            total_loss_test+=a
            model_loss_test+=b
            rpn_loss_cls_test+=c
            side_loss_test+=d
            v_loss_test+=e
            o_loss_test+=f

        total_loss_test/=sample_num
        model_loss_test/=sample_num
        rpn_loss_cls_test/=sample_num
        side_loss_test/=sample_num
        v_loss_test/=sample_num
        o_loss_test/=sample_num

        return total_loss_test,model_loss_test, rpn_loss_cls_test, side_loss_test, v_loss_test, o_loss_test



    def train_model(self, sess, training_flag, max_iters, restore=False):
        """Network training loop."""
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        data_layer_test = get_data_layer(self.roidb_test, self.imdb_test.num_classes)
        total_loss, model_loss, rpn_cross_entropy, side_loss, v_loss, o_loss, fg_num, bg_num, fg_loss, bg_loss, side_label, side_prob =self.net.build_loss_hx4(ohem=cfg.TRAIN.OHEM)
        # scalar summary
        tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
        tf.summary.scalar('v_loss', v_loss)
        tf.summary.scalar('o_loss', o_loss)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss',total_loss)
        summary_op = tf.summary.merge_all()

        log_image, log_image_data, log_image_name =self.build_image_summary()

        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load vgg16
        if self.pretrained_model is not None and not restore :
            try:
                print(('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model))
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise Exception('Check your pretrained model {:s}'.format(self.pretrained_model))

        # resuming a trainer
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        last_snapshot_iter = -1
        timer = Timer()
        big_loss_img_name = list()
        for iter in range(restore_iter, max_iters):
            timer.tic()
            # learning rate
            if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
                sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
                print(lr)

            # get one batch
            db_inds,blobs = data_layer.forward()

            # if blobs['im_name']!='auto_50_5768038962_20180628224921_20180629100000_161.jpg':
            #     continue

            # for densenet
            feed_dict={
                self.net.data: blobs['data'],
                self.net.img_name: blobs['im_name'],
                self.net.im_info: blobs['im_info'],
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: blobs['gt_boxes'],
                self.net.gt_ishard: blobs['gt_ishard'],
                self.net.dontcare_areas: blobs['dontcare_areas'],
                training_flag: True
            }
            res_fetches=[]
            fetch_list = [total_loss, model_loss, rpn_cross_entropy, side_loss, v_loss,o_loss, fg_num, bg_num, fg_loss, bg_loss, side_label, side_prob,
                          summary_op,
                          train_op] + res_fetches



            for i in range(2):
                total_loss_val, model_loss_val, rpn_loss_cls_val, side_loss_val, v_loss_val, o_loss_val, fg_num_val, bg_num_val, fg_loss_val, bg_loss_val, side_label_val, side_prob_val, \
                summary_str, _ = sess.run(fetches=fetch_list, feed_dict=feed_dict)

                # print(
                #     'iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, side_loss_: %.4f, v_loss: %.4f, o_loss: %.4f, fg_num: %.4f, bg_num: %.4f, fg_loss: %.4f, bg_loss: %.4f,lr: %f' % \
                #     (iter, max_iters, total_loss_val, model_loss_val, rpn_loss_cls_val, side_loss_val, v_loss_val,
                #      o_loss_val,
                #      fg_num_val, bg_num_val, fg_loss_val, bg_loss_val, lr.eval()))

                assert np.sum(total_loss_val != total_loss_val) == 0, "出现了nan损失，图片名字是" + str(blobs['im_name'])
                if rpn_loss_cls_val > 1:
                    big_loss_img_name.append(blobs['im_name'])



            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            _diff_time = timer.toc(average=False)



            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                # for test
                total_loss_test, model_loss_test, rpn_loss_cls_test, side_loss_test, v_loss_test, o_loss_test = self.get_test_loss(sess,
                                                                                                            training_flag,
                                                                                                            data_layer_test,
                                                                                                            total_loss,
                                                                                                            model_loss,
                                                                                                            rpn_cross_entropy,
                                                                                                            side_loss,
                                                                                                            v_loss,
                                                                                                            o_loss)

                print(
                    'iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, side_loss_: %.4f, v_loss: %.4f, o_loss: %.4f, fg_num: %.4f, bg_num: %.4f, fg_loss: %.4f, bg_loss: %.4f,lr: %f' % \
                    (iter, max_iters, total_loss_val, model_loss_val, rpn_loss_cls_val, side_loss_val, v_loss_val,
                     o_loss_val,
                     fg_num_val, bg_num_val, fg_loss_val, bg_loss_val, lr.eval()))
                print(
                    'iter: %d / %d, total test loss: %.4f, test model loss: %.4f, test rpn_loss_cls: %.4f, test_side_loss: %.4f,  test v_loss: %.4f, test o_loss: %.4f, lr: %f' % \
                    (iter, max_iters, total_loss_test, model_loss_test, rpn_loss_cls_test, side_loss_test, v_loss_test, o_loss_test, lr.eval()))
                print('speed: {:.3f}s / iter'.format(_diff_time))

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

        with open(os.path.join(cfg.ROOT_DIR, 'error','big_cls_loss_img.txt'), 'w') as f:
            for a in big_loss_img_name:
                f.write(str(a)+'\n')


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED and 0:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    if cfg.TRAIN.HAS_RPN:
            rdl_roidb.prepare_roidb(imdb)
    else:
        rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def get_test_roidb(imdb_test):
    """Returns a roidb (Region of Interest database) for use in test."""
    print('Preparing test data...')
    if cfg.TRAIN.HAS_RPN:
        rdl_roidb.prepare_roidb(imdb_test)
    else:
        rdl_roidb.prepare_roidb(imdb_test)
    print('done')

    return imdb_test.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            # obsolete
            # layer = GtDataLayer(roidb)
            raise "Calling caffe modules..."
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def train_net(network, training_flag, imdb, roidb, imdb_test, roidb_test, output_dir, log_dir, pretrained_model=None, max_iters=40000, restore=False):
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, imdb_test, roidb_test, output_dir, logdir= log_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, training_flag, max_iters, restore=restore)
        print('done solving')
