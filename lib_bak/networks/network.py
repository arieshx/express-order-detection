# -*- coding:utf-8 -*-
import numpy as np, os
import tensorflow as tf
from lib.fast_rcnn.config import cfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from lib.rpn_msr.proposal_layer_hx import proposal_layer as proposal_layer_py_hx


from lib.rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from lib.rpn_msr.anchor_target_layer_hx import anchor_target_layer as anchor_target_layer_py_hx
from lib.rpn_msr.anchor_target_layer_hx2 import anchor_target_layer as anchor_target_layer_py_hx2
from lib.rpn_msr.anchor_target_layer_hx3 import anchor_target_layer as anchor_target_layer_py_hx3
DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_path = os.path.join(cfg.ROOT_DIR, data_path)
        data_dict = np.load(data_path,encoding='latin1').item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in list(self.layers.items()))+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


    @layer
    def Bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell, img, dtype=tf.float32)
            lstm_out = tf.concat(lstm_out, axis=-1)

            lstm_out = tf.reshape(lstm_out, [N * H * W, 2*d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [2*d_h, d_o], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def lstm(self, input, d_i,d_h,d_o, name, trainable=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N,H,W,C = shape[0], shape[1],shape[2], shape[3]
            img = tf.reshape(img,[N*H,W,C])
            img.set_shape([None,None,d_i])

            lstm_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            initial_state = lstm_cell.zero_state(N*H, dtype=tf.float32)

            lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, img,
                                               initial_state=initial_state,dtype=tf.float32)

            lstm_out = tf.reshape(lstm_out,[N*H*W,d_h])


            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [d_h, d_o], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases


            outputs = tf.reshape(outputs, [N,H,W,d_o])
            return outputs

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            input = tf.reshape(input, [N*H*W,C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [d_i, d_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope(name) as scope:
            blob,bbox_delta = tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales],\
                                     [tf.float32,tf.float32])

            rpn_rois = tf.convert_to_tensor(tf.reshape(blob,[-1, 5]), name = 'rpn_rois') # shape is (1 x H x W x A, 2)
            rpn_targets = tf.convert_to_tensor(bbox_delta, name = 'rpn_targets') # shape is (1 x H x W x A, 4)
            self.layers['rpn_rois'] = rpn_rois
            self.layers['rpn_targets'] = rpn_targets

            return rpn_rois, rpn_targets

    @layer
    def proposal_layer_hx(self, input, _feat_stride, anchor_scales, cfg_key, name):
        """
        实现论文中的用三个回归值获取proposal
        :param input:
        :param _feat_stride:
        :param anchor_scales:
        :param cfg_key:
        :param name:
        :return:
        """
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope(name) as scope:
            blob, bbox_v, bbox_o = tf.py_func(proposal_layer_py_hx,
                                          [input[0], input[1], input[2],input[3], cfg_key, _feat_stride, anchor_scales], \
                                          [tf.float32, tf.float32, tf.float32])

            rpn_rois = tf.convert_to_tensor(tf.reshape(blob, [-1, 5]), name='rpn_rois')  # shape is (1 x H x W x A, 2)
            bbox_v = tf.convert_to_tensor(tf.reshape(bbox_v, [-1, 2]), name='bbox_v')
            bbox_o = tf.convert_to_tensor(tf.reshape(bbox_o, [-1]), name='bbox_o')
            #rpn_targets = tf.convert_to_tensor(bbox_delta, name='rpn_targets')  # shape is (1 x H x W x A, 4)
            self.layers['rpn_rois'] = rpn_rois
            self.layers['rpn_bbox_v'] = bbox_v
            self.layers['rpn_bbox_o'] = bbox_o
            #self.layers['rpn_targets'] = rpn_targets

            return rpn_rois, bbox_v, bbox_o

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0],input[1],input[2],input[3],input[4], _feat_stride, anchor_scales],
                           [tf.float32,tf.float32,tf.float32,tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels') # shape is (1 x H x W x A, 2)
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets') # shape is (1 x H x W x A, 4)
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights') # shape is (1 x H x W x A, 4)
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights') # shape is (1 x H x W x A, 4)


            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def anchor_target_layer_hx(self, input, _feat_stride, anchor_scales, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
            rpn_labels, rpn_v_target, rpn_o_target, j_index, k_index = \
                tf.py_func(anchor_target_layer_py_hx3,
                           [input[0], input[1], input[2], input[3], input[4], input[5], input[6],  _feat_stride, anchor_scales],
                           [tf.float32, tf.float32, tf.float32, tf.int32, tf.int32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                              name='rpn_labels')  # shape is (1 x H x W x A, 2)
            rpn_v_target = tf.convert_to_tensor(rpn_v_target, name='rpn_v_target')  # shape is (1 x H x W x A, 4)
            rpn_o_target = tf.convert_to_tensor(rpn_o_target, name='rpn_o_target')  # shape is (1 x H x W x A, 4)
            j_index = tf.convert_to_tensor(j_index, name='j_index')  # shape is (1 x H x W x A, 4)
            k_index = tf.convert_to_tensor(k_index, name='k_index')

            return rpn_labels, rpn_v_target, rpn_o_target, j_index, k_index

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input,\
                               [input_shape[0],\
                                input_shape[1], \
                                -1,\
                                int(d)])


    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1])

    @layer
    def batch_normalization(self,input,name,relu=True,is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                #return tf.mul(l2_weight, tf.nn.l2_loss(tensor), name='value')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)



    def build_loss(self, ohem=False):
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)
        # ignore_label(-1)
        fg_keep = tf.equal(rpn_label, 1)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)  # shape (N, 2)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label,logits=rpn_cls_score)

        # box loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax4)
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)
        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)


        model_loss = rpn_cross_entropy +  rpn_loss_box

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss,model_loss, rpn_cross_entropy, rpn_loss_box

    def build_loss_hx(self, ohem=False, lambda1 =1, lambda2 = 2):
        """
        实现论文中的损失函数，包括side-refinement
        :return:
        """
        rpn_data = self.get_output('rpn-data')

        # loss1 分类误差
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(rpn_data[0], [-1])  # shape (HxWxA)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))  # (256, 1) 损失1 的索引，就是说所有的256个anchor的索引,

        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n) # 求均值
        # 对分类误差显示facal loss
        rpn_cls_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])  # shape (HxWxA, 2)


        # loss2 垂直方向位置的回归
        j_index1 = tf.reshape(rpn_data[3], [-1])  # (j,)
        j_index = tf.where(tf.equal(j_index1,1))  # (j, 1)
        v_target = rpn_data[1]  # (H*W*A, 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax2) #修改以后只要回归两个值
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1,2]), j_index)   #
        v_target = tf.gather(tf.reshape(v_target, [-1,2]), j_index)
        j_index_num = (tf.reduce_sum(tf.cast(j_index1, tf.float32)) + 1)
        v_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_bbox_pred-v_target))/j_index_num    # 我感觉这个也可以求均值

        # loss3 side refinement
        k_index1 = tf.reshape(rpn_data[4], [-1])
        k_index = tf.where(tf.equal(k_index1,1))
        o_target = rpn_data[2]  # (H*W*A, )
        rpn_side_refinement = self.get_output('rpn_side_refinement')  #
        rpn_side_refinement = tf.gather(tf.reshape(rpn_side_refinement, [-1]), k_index)
        o_target = tf.gather(tf.reshape(o_target, [-1]), k_index)
        k_index_num = tf.reduce_sum(tf.cast(k_index1, tf.float32)) + 1
        o_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_side_refinement - o_target))/k_index_num

        model_loss = rpn_cross_entropy + lambda1*v_loss + lambda2*o_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss,rpn_cross_entropy, v_loss, o_loss

    def build_loss_hx2(self, ohem=False, lambda1 =1, lambda2 = 2):
        """
        f分类误差使用dice loss，所有正负样本都参与训练，结果是模型会被负样本支配，no work
        :return:
        """
        rpn_data = self.get_output('rpn-data')

        # loss1 分类误差
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(rpn_data[0], [-1])  # shape (HxWxA)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))  # (256, 1) 损失1 的索引，就是说所有的256个anchor的索引,

        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
        rpn_label = tf.reshape(rpn_label, [-1])
        # 改进分类误差：想要使正负负样本都能够参与到分类中，由于正负样本严重不均衡，之前的随机抽取样本使均衡的结果不好的地方是，边框不准，会把负样本检测成正样本。
        fg = tf.equal(rpn_label, 1)
        bg = tf.equal(rpn_label, 0)
        beata = 1 - tf.reduce_sum(tf.cast(fg, tf.float32))*1./(tf.reduce_sum(tf.cast(fg, tf.float32)) + tf.reduce_sum(tf.cast(bg, tf.float32)))
        prob_pre = rpn_cls_score[:, 1]  # (HxWxA, 1)
        prob_pre = tf.reshape(prob_pre, [-1])  # (HxWxA)
        rpn_label = tf.cast(rpn_label, tf.float32)
        eps = 1e-5
        intersection = tf.reduce_sum(rpn_label * prob_pre)
        union = tf.reduce_sum(rpn_label) + tf.reduce_sum(prob_pre) + eps
        dice_loss = 1. - (2 * intersection / union)
        rpn_cross_entropy = dice_loss


        # loss2 垂直方向位置的回归
        j_index1 = tf.reshape(rpn_data[3], [-1])  # (j,)
        j_index = tf.where(tf.equal(j_index1,1))  # (j, 1)
        v_target = rpn_data[1]  # (H*W*A, 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax2) #修改以后只要回归两个值
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1,2]), j_index)   #
        v_target = tf.gather(tf.reshape(v_target, [-1,2]), j_index)
        j_index_num = (tf.reduce_sum(tf.cast(j_index1, tf.float32)) + 1)
        v_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_bbox_pred-v_target))/j_index_num    # 我感觉这个也可以求均值

        # loss3 side refinement
        k_index1 = tf.reshape(rpn_data[4], [-1])
        k_index = tf.where(tf.equal(k_index1,1))
        o_target = rpn_data[2]  # (H*W*A, )
        rpn_side_refinement = self.get_output('rpn_side_refinement')  #
        rpn_side_refinement = tf.gather(tf.reshape(rpn_side_refinement, [-1]), k_index)
        o_target = tf.gather(tf.reshape(o_target, [-1]), k_index)
        k_index_num = tf.reduce_sum(tf.cast(k_index1, tf.float32)) + 1
        o_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_side_refinement - o_target))/k_index_num

        model_loss = rpn_cross_entropy + lambda1*v_loss + lambda2*o_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss,rpn_cross_entropy, v_loss, o_loss


    def build_loss_hx3(self, ohem=False, lambda1 =1, lambda2 = 2):
        """
        每一张图，第一遍正负随机生成正负样本，第二遍包括了挖掘的难样本，挖掘难样本以后，正负比例依然失衡，难易程度没有衡量，因此加入了facal loss
        实验结果：前景框都检测为了背景，说明模型可能被负样本支配，挖掘难样本以后正负样本比例依然相当不平衡，1:10这样子,
        改进，OHEM + facal loss 待改进
        :return:
        """
        rpn_data = self.get_output('rpn-data')

        # loss1 分类误差
        # 对分类误差显示facal loss，使用ate = 0.25或者beata，
        rpn_cls_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(rpn_data[0], [-1])  # shape (HxWxA)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_prob = tf.gather(rpn_cls_prob, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 2])
        rpn_label = tf.reshape(rpn_label, [-1])
        # 求一个负样本的比例
        fg = tf.equal(rpn_label, 1)
        bg = tf.equal(rpn_label, 0)
        ate = 0.25
        beata = 1 - tf.reduce_sum(tf.cast(fg, tf.float32)) * 1. / (
                    tf.reduce_sum(tf.cast(fg, tf.float32)) + tf.reduce_sum(tf.cast(bg, tf.float32)))
        fg_prob = rpn_cls_prob[:, 1]  # (HxWxA, 1)
        fg_prob = tf.reshape(fg_prob, [-1])  # (HxWxA)
        rpn_label = tf.cast(rpn_label, tf.float32)
        loss1 = -(tf.pow(1-fg_prob, 2)*rpn_label*tf.log(fg_prob) + tf.pow(fg_prob, 2)*(1-rpn_label)*tf.log(1-fg_prob))
        loss1 = -(1./(1-beata))*(beata*tf.pow(1-fg_prob, 2)*rpn_label*tf.log(fg_prob) + (1-beata)*tf.pow(fg_prob, 2)*(1-rpn_label)*tf.log(1-fg_prob))
        rpn_cross_entropy = tf.reduce_mean(loss1)

        # loss2 垂直方向位置的回归
        j_index1 = tf.reshape(rpn_data[3], [-1])  # (j,)
        j_index = tf.where(tf.equal(j_index1,1))  # (j, 1)
        v_target = rpn_data[1]  # (H*W*A, 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax2) #修改以后只要回归两个值
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1,2]), j_index)   #
        v_target = tf.gather(tf.reshape(v_target, [-1,2]), j_index)
        j_index_num = (tf.reduce_sum(tf.cast(j_index1, tf.float32)) + 1)
        v_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_bbox_pred-v_target))/j_index_num    # 我感觉这个也可以求均值

        # loss3 side refinement
        k_index1 = tf.reshape(rpn_data[4], [-1])
        k_index = tf.where(tf.equal(k_index1,1))
        o_target = rpn_data[2]  # (H*W*A, )
        rpn_side_refinement = self.get_output('rpn_side_refinement')  #
        rpn_side_refinement = tf.gather(tf.reshape(rpn_side_refinement, [-1]), k_index)
        o_target = tf.gather(tf.reshape(o_target, [-1]), k_index)
        k_index_num = tf.reduce_sum(tf.cast(k_index1, tf.float32)) + 1
        o_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_side_refinement - o_target))/k_index_num

        model_loss = rpn_cross_entropy + lambda1*v_loss + lambda2*o_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss,rpn_cross_entropy, v_loss, o_loss


    def build_loss_hx4(self, ohem=False, lambda1 =1, lambda2 = 2):
        """
        方法：纯facal loss.所有正负样本都参与，归一化用正样本数量去归一化
        :return:
        """
        rpn_data = self.get_output('rpn-data')

        # loss1 分类误差
        # 对分类误差显示facal loss，使用ate = 0.25或者beata，
        rpn_cls_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(rpn_data[0], [-1])  # shape (HxWxA)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_prob = tf.gather(rpn_cls_prob, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 2])
        rpn_label = tf.reshape(rpn_label, [-1])
        # 求一个负样本的比例
        fg = tf.equal(rpn_label, 1)
        bg = tf.equal(rpn_label, 0)
        ate = 0.7
        beata = 1 - tf.reduce_sum(tf.cast(fg, tf.float32)) * 1. / (
                    tf.reduce_sum(tf.cast(fg, tf.float32)) + tf.reduce_sum(tf.cast(bg, tf.float32)))
        fg_prob = rpn_cls_prob[:, 1]  # (HxWxA, 1)
        fg_prob = tf.reshape(fg_prob, [-1])  # (HxWxA)
        rpn_label = tf.cast(rpn_label, tf.float32)
        fg_loss = -ate*tf.pow(1-fg_prob, 2)*rpn_label*tf.log(fg_prob)
        bg_loss = -(1-ate)*tf.pow(fg_prob, 2)*(1-rpn_label)*tf.log(1-fg_prob)
        # fg_loss = - rpn_label * tf.log(fg_prob)
        # bg_loss = -(1 - rpn_label) * tf.log(1 - fg_prob)
        loss1 = fg_loss + bg_loss
        fg_num = tf.reduce_sum(tf.cast(fg, tf.float32))+1
        bg_num = tf.reduce_sum(tf.cast(bg, tf.float32))+1
        fg_loss = tf.reduce_sum(fg_loss)
        bg_loss = tf.reduce_sum(bg_loss)
        rpn_cross_entropy = tf.reduce_sum(loss1)/fg_num
        # 在论文的基础loss上加上一份针对边界的分类误差，这样子有点不美观，之后将分类误差统一到loss1中，思路是：对我们额外关注的边界anchor的分类加权重
        side_index = tf.reshape(rpn_data[4], [-1])
        side_index = tf.where(tf.equal(side_index, 1))
        side_label = tf.reshape(rpn_data[0], [-1])
        side_label = tf.gather(side_label, side_index)
        side_label = tf.reshape(side_label, [-1])
        side_label = tf.cast(side_label, tf.float32)
        side_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])
        side_prob = tf.gather(side_prob, side_index)
        side_prob = tf.reshape(side_prob, [-1, 2])
        side_prob = side_prob[:, 1]
        side_prob = tf.reshape(side_prob, [-1])
        side_loss = -(side_label*tf.log(side_prob) + (1 - side_label) * tf.log(1 - side_prob))
        side_loss = tf.reduce_mean(side_loss)

        # loss2 垂直方向位置的回归
        j_index1 = tf.reshape(rpn_data[3], [-1])  # (j,)
        j_index = tf.where(tf.equal(j_index1,1))  # (j, 1)
        v_target = rpn_data[1]  # (H*W*A, 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax2) #修改以后只要回归两个值
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1,2]), j_index)   #
        v_target = tf.gather(tf.reshape(v_target, [-1,2]), j_index)
        j_index_num = (tf.reduce_sum(tf.cast(j_index1, tf.float32)) + 1)
        v_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_bbox_pred-v_target))/j_index_num    # 我感觉这个也可以求均值

        # loss3 side refinement
        k_index1 = tf.reshape(rpn_data[4], [-1])
        k_index = tf.where(tf.equal(k_index1,1))
        o_target = rpn_data[2]  # (H*W*A, )
        rpn_side_refinement = self.get_output('rpn_side_refinement')  #
        rpn_side_refinement = tf.gather(tf.reshape(rpn_side_refinement, [-1]), k_index)
        o_target = tf.gather(tf.reshape(o_target, [-1]), k_index)
        k_index_num = tf.reduce_sum(tf.cast(k_index1, tf.float32)) + 1
        o_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_side_refinement - o_target))/k_index_num

        model_loss = rpn_cross_entropy + lambda1*v_loss + lambda2*o_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss,rpn_cross_entropy, side_loss, v_loss, o_loss, fg_num, bg_num, fg_loss, bg_loss, side_label, side_prob

    def build_loss_hx5(self, ohem=False, lambda1 =1, lambda2 = 2):
        """
        方法：纯facal loss.所有正负样本都参与，归一化用正样本数量去归一化
        额外trick:对我们额外关注的边界框分类增加额外权重，
        :return:
        """
        rpn_data = self.get_output('rpn-data')

        # loss1 分类误差
        # 对分类误差显示facal loss，使用ate = 0.25或者beata，
        rpn_cls_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])  # shape (HxWxA, 2)
        rpn_label = tf.reshape(rpn_data[0], [-1])  # shape (HxWxA)
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        rpn_cls_prob = tf.gather(rpn_cls_prob, rpn_keep)
        rpn_label = tf.gather(rpn_label, rpn_keep)
        rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 2])
        rpn_label = tf.reshape(rpn_label, [-1])
        # 求一个负样本的比例
        fg = tf.equal(rpn_label, 1)
        bg = tf.equal(rpn_label, 0)
        ate = 0.7
        beata = 1 - tf.reduce_sum(tf.cast(fg, tf.float32)) * 1. / (
                    tf.reduce_sum(tf.cast(fg, tf.float32)) + tf.reduce_sum(tf.cast(bg, tf.float32)))
        fg_prob = rpn_cls_prob[:, 1]  # (HxWxA, 1)
        fg_prob = tf.reshape(fg_prob, [-1])  # (HxWxA)
        rpn_label = tf.cast(rpn_label, tf.float32)
        fg_loss = -ate*tf.pow(1-fg_prob, 2)*rpn_label*tf.log(fg_prob)
        bg_loss = -(1-ate)*tf.pow(fg_prob, 2)*(1-rpn_label)*tf.log(1-fg_prob)
        # fg_loss = - rpn_label * tf.log(fg_prob)
        # bg_loss = -(1 - rpn_label) * tf.log(1 - fg_prob)
        loss1 = fg_loss + bg_loss
        fg_num = tf.reduce_sum(tf.cast(fg, tf.float32))+1
        bg_num = tf.reduce_sum(tf.cast(bg, tf.float32))+1
        fg_loss = tf.reduce_sum(fg_loss)
        bg_loss = tf.reduce_sum(bg_loss)
        rpn_cross_entropy = tf.reduce_sum(loss1)/fg_num
        # 在论文的基础loss上加上一份针对边界的分类误差，这样子有点不美观，之后将分类误差统一到loss1中，思路是：对我们额外关注的边界anchor的分类加权重
        side_index = tf.reshape(rpn_data[4], [-1])
        side_index = tf.where(tf.equal(side_index, 1))
        side_label = tf.reshape(rpn_data[0], [-1])
        side_label = tf.gather(side_label, side_index)
        side_label = tf.reshape(side_label, [-1])
        side_label = tf.cast(side_label, tf.float32)
        side_prob = tf.reshape(self.get_output('rpn_cls_prob'), [-1, 2])
        side_prob = tf.gather(side_prob, side_index)
        side_prob = tf.reshape(side_prob, [-1, 2])
        side_prob = side_prob[:, 1]
        side_prob = tf.reshape(side_prob, [-1])
        side_loss = -(side_label*tf.log(side_prob) + (1 - side_label) * tf.log(1 - side_prob))
        side_loss = tf.reduce_mean(side_loss)

        # loss2 垂直方向位置的回归
        j_index1 = tf.reshape(rpn_data[3], [-1])  # (j,)
        j_index = tf.where(tf.equal(j_index1,1))  # (j, 1)
        v_target = rpn_data[1]  # (H*W*A, 2)
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape (1, H, W, Ax2) #修改以后只要回归两个值
        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1,2]), j_index)   #
        v_target = tf.gather(tf.reshape(v_target, [-1,2]), j_index)
        j_index_num = (tf.reduce_sum(tf.cast(j_index1, tf.float32)) + 1)
        v_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_bbox_pred-v_target))/j_index_num    # 我感觉这个也可以求均值

        # loss3 side refinement
        k_index1 = tf.reshape(rpn_data[4], [-1])
        k_index = tf.where(tf.equal(k_index1,1))
        o_target = rpn_data[2]  # (H*W*A, )
        rpn_side_refinement = self.get_output('rpn_side_refinement')  #
        rpn_side_refinement = tf.gather(tf.reshape(rpn_side_refinement, [-1]), k_index)
        o_target = tf.gather(tf.reshape(o_target, [-1]), k_index)
        k_index_num = tf.reduce_sum(tf.cast(k_index1, tf.float32)) + 1
        o_loss = tf.reduce_sum(self.smooth_l1_dist(rpn_side_refinement - o_target))/k_index_num

        model_loss = rpn_cross_entropy + side_loss + lambda1*v_loss + lambda2*o_loss

        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        return total_loss, model_loss,rpn_cross_entropy, side_loss, v_loss, o_loss, fg_num, bg_num, fg_loss, bg_loss, side_label, side_prob