import tensorflow as tf
import numpy as np
import sys
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("log_dir", "./logs", "log dir.")
tf.app.flags.DEFINE_string("model_dir", "./models", "model dir.")
tf.app.flags.DEFINE_string("query_embedding", "./query_embedding", "query embedding file.")
tf.app.flags.DEFINE_string("data", "./head.test", "data file.")
tf.app.flags.DEFINE_string("val_data", "./head.test", "data file.")
tf.app.flags.DEFINE_string("fc_dims", "1024", "node counts seperated by comma.")
tf.app.flags.DEFINE_integer("em_len", 100, "embedding length.")
tf.app.flags.DEFINE_string("devices", "0,1", "available devices.")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 100, "epoch num.")

class FmModel(object):
    def __init__(self, feature_len, latent_len):
        self.latent_len = latent_len
        self.feature_len = feature_len
        self.em_len = FLAGS.em_len

        self.X = tf.placeholder('float', shape=[None, self.feature_len])
        self.y = tf.placeholder('float', shape=[None, 1])

        # Embedding.
        # self.U = tf.placeholder('float', shape=[None, self.em_len])
        # self.Q = tf.placeholder('float', shape=[None, self.em_len])
        # self.uq = tf.concat(axis=1, values=[self.U, self.Q])
        # self.sim = tf.reduce_sum(tf.multiply(self.U, self.Q), 1, keep_dims=True)

        self.U = tf.placeholder('int32', shape=[FLAGS.batch_size])
        self.Q = tf.placeholder('int32', shape=[FLAGS.batch_size])
        self.u_embeddings = tf.Variable(tf.random_uniform([USER_COUNT, self.em_len], -1.0, 1.0))
        self.q_embeddings = tf.Variable(tf.random_uniform([QUERY_COUNT, self.em_len], -1.0, 1.0))
        self.embed_u = tf.nn.embedding_lookup(self.u_embeddings, self.U)
        self.embed_q = tf.nn.embedding_lookup(self.q_embeddings, self.Q)
        self.uq = tf.concat(axis=1, values=[self.embed_u, self.embed_q])
        self.sim = tf.reduce_sum(tf.multiply(self.embed_u, self.embed_q), 1, keep_dims=True)

        if FLAGS.fc_dims:
            fc_dim_list = FLAGS.fc_dims.split(',')
            input_layer = self.uq
            for fc_dim in fc_dim_list:
                fc_dim = int(fc_dim)
                output_layer = tf.layers.dense(inputs=input_layer, units=fc_dim,
                                               activation=tf.nn.relu)
                input_layer = output_layer

            self.fc_dim = int(fc_dim_list[-1])
            self.fc_out = output_layer
        else:
            self.fc_dim = self.em_len * 2
            self.fc_out = self.uq

        # Concat features, fc, sim
        self.middle = tf.concat(axis=1, values=[self.X, self.fc_out, self.sim])
        self.middle_len = self.feature_len + self.fc_dim + 1

        # Bias and weights
        self.b = tf.Variable(tf.zeros([1]))
        self.w_matrix = tf.Variable(tf.zeros([self.middle_len]))
        linear_terms = tf.add(
            self.b,
            tf.reduce_sum(tf.multiply(self.w_matrix, self.middle),
                          1,
                          keep_dims=True))

        # Interaction factors, randomly initialized
        self.v_matrix = tf.Variable(tf.random_normal([self.latent_len, self.middle_len],
                                                     stddev=0.01))
        pair_interactions = tf.multiply(
            0.5,
            tf.reduce_sum(
                tf.subtract(
                    tf.pow(tf.matmul(self.middle, tf.transpose(self.v_matrix)), 2),
                    tf.matmul(tf.pow(self.middle, 2), tf.transpose(tf.pow(self.v_matrix, 2)))),
                1,
                keep_dims=True))

        self.pred = tf.add(linear_terms, pair_interactions)

        self.lambda_w = tf.constant(0.001, name='lambda_w')
        self.lambda_v = tf.constant(0.001, name='lambda_v')
        self.l2_norm = tf.reduce_sum(
            tf.add(
                tf.multiply(self.lambda_w, tf.pow(self.w_matrix, 2)),
                tf.multiply(self.lambda_v, tf.pow(self.v_matrix, 2))))

        self.error = tf.losses.hinge_loss(self.y, self.pred)
        #self.error = tf.reduce_mean(tf.square(tf.subtract(self.y, self.pred)))
        self.loss = tf.add(self.error, tf.multiply(0.2, self.l2_norm))

        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        gpu_options = tf.GPUOptions(allow_growth=True,
                                    visible_device_list=FLAGS.devices)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                     allow_soft_placement=True))

        self.auc_score = tf.metrics.auc(self.y, tf.nn.softmax(self.pred))

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def batch(self, X_, U_, Q_, y_, batch_size=-1):
        n_samples = X_.shape[0]

        if batch_size == -1:
            batch_size = n_samples
        if batch_size < 1:
            raise ValueError('batch_size should be >= 1.')

        for i in range(0, n_samples, batch_size):
            upper_bound = min(i + batch_size, n_samples)
            ret_x = X_[i:upper_bound]
            ret_u = U_[i:upper_bound]
            ret_q = Q_[i:upper_bound]
            ret_y = None
            if y_ is not None:
                ret_y = y_[i:i + batch_size]
                yield (ret_x.toarray(), ret_u, ret_q, ret_y)

    def fit(self, train_data, val_data=None):
        X_train, U_train, Q_train, y_train = train_data
        do_val = False
        if not val_data is None:
            X_val, U_val, Q_val, y_val = val_data
            do_val = True

        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())
        self.sess.run(init)
        for epoch in xrange(FLAGS.epoch):
            perm = np.random.permutation(X_train.shape[0])
            total_batch_count = int(X_train.shape[0] / FLAGS.batch_size)
            print 'Epoch %d.' % (epoch + 1)
            error = 0
            auc = 0
            batch_count = 0
            for bX, bU, bQ, bY in self.batch(X_train[perm], U_train[perm],
                                             Q_train[perm], y_train[perm],
                                             FLAGS.batch_size):
                if bX.shape[0] != FLAGS.batch_size:
                    continue

                _, preds, c_error = self.sess.run(
                    [self.optimizer, self.pred, self.error],
                    feed_dict={self.X: bX.reshape(-1, self.feature_len),
                               self.U: bU,
                               self.Q: bQ,
                               self.y: bY.reshape(-1, 1)})
                error += c_error
                auc += average_precision_score(bY, preds)
                batch_count += 1
                sys.stderr.write('%05d/%d\r' % (batch_count,
                                                total_batch_count))

            train_error = error / batch_count
            train_auc = auc / batch_count
            if epoch % 10 == 0:
                self.save('%s/%d-model' % (FLAGS.model_dir, epoch + 1))

            if not do_val:
                sys.stderr.write('Training error: %f, training auc.\n' %
                                 (train_error, train_auc))
                continue

            val_error = 0
            val_auc = 0
            val_count = 0
            for bX, bU, bQ, bY in self.batch(X_val, U_val, Q_val, y_val,
                                             FLAGS.batch_size):
                if bX.shape[0] != FLAGS.batch_size:
                    continue

                c_error, preds = self.sess.run(
                    [self.error, self.pred],
                    feed_dict={self.X: bX.reshape(-1, self.feature_len),
                               self.U: bU,
                               self.Q: bQ,
                               self.y: bY.reshape(-1, 1)})
                val_auc += average_precision_score(bY, preds)
                val_error += c_error
                val_count += 1
            val_error /= val_count
            val_auc /= val_count
            sys.stderr.write('Training error: %f, training auc: %f,'
                             'Val error: %f, val auc: %f.\n' %
                             (train_error, train_auc, val_error, val_auc))


if __name__ == '__main__':
    from analyze_embedding import load_query_embedding, process_id
    #query_embedding_dict = load_query_embedding(FLAGS.query_embedding)
    user_dict = {}
    query_dict = {}
    train_X, train_U, train_Q, train_y, user_dict, query_dict = process_id(FLAGS.data, user_dict, query_dict)
    val_X, val_U, val_Q, val_y, user_dict, query_dict = process_id(FLAGS.val_data, user_dict, query_dict)
    USER_COUNT = len(user_dict) + 1
    QUERY_COUNT = len(query_dict) + 1

    #print train_X.shape, train_U.shape, train_Q.shape
    model = FmModel(train_X.shape[1], 10)
    model.fit((train_X, train_U, train_Q, train_y),
              (val_X, val_U, val_Q, val_y))
