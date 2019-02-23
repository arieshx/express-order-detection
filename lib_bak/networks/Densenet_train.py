# -*- coding:utf-8 -*-
import tensorflow as tf
from network import Network
from lib.fast_rcnn.config import cfg
from densenet import *

class densenet_train(Network):
    def __init__(self,training_flag, nb_blocks = 2, filters = k, trainable=True, ):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='gt_boxes')
        self.gt_ishard = tf.placeholder(tf.int32, shape=[None], name='gt_ishard')
        self.dontcare_areas = tf.placeholder(tf.float32, shape=[None, 4], name='dontcare_areas')
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes, \
                            'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.trainable = trainable

        #

        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training_flag
        self.setup()

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        """
        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))
        """

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        # x = Relu(x)
        # x = Global_Average_Pooling(x)
        # x = flatten(x)
        # x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x

    def setup(self):
        anchor_scales = cfg.ANCHOR_SCALES
        _feat_stride = [16, ]
        anchor_num = cfg.ANCHOR_NUM

        bn = self.Dense_net(self.data)

        x = Relu(bn)
        self.layers['relu_dense'] = x
        self.inputs = [x]

        (self.feed('relu_dense').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))

        (self.feed('rpn_conv/3x3').Bilstm(512, 128, 512, name='lstm_o'))
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * anchor_num * 2, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512, len(anchor_scales) * anchor_num * 2, name='rpn_cls_score'))
        (self.feed('lstm_o')
         .lstm_fc(512, len(anchor_scales) * anchor_num * 1, name='rpn_side_refinement'))

        # generating training labels on the fly
        # output: rpn_labels(HxWxA, 2) rpn_bbox_targets(HxWxA, 4) rpn_bbox_inside_weights rpn_bbox_outside_weights
        # 给每个anchor上标签，并计算真值（也是delta的形式），以及内部权重和外部权重
        (self.feed('rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info')
         .anchor_target_layer_hx(_feat_stride, anchor_scales, name='rpn-data'))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))



if __name__ == "__main__":
    training = tf.placeholder(tf.bool)
    model = densenet_train(training,trainable=False)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        import os
        save_log_sir = os.path.join(cfg.ROOT_DIR,'logs_densenet')
        summary_writer = tf.summary.FileWriter(save_log_sir, sess.graph)