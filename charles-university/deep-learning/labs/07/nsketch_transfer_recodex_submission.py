# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import nets.nasnet.nasnet

class Dataset:
    def __init__(self, filename, shuffle_batches = True):
        data = np.load(filename)
        self._images = data[\"images\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
            len(self._images))
        # Normalize images
        # self._images = (self._images - self._images.mean(axis=0)) / (self._images.std(axis=0))

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._images[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._images)) if self._shuffle_batches else np.arange(
                len(self._images))
            return True
        return False

    def batches(self, batch_size, shift_fraction=0.):
        x, y = self._images, self._labels

        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=shift_fraction, height_shift_range=shift_fraction)
        gen = train_datagen.flow(x, y, batch_size=batch_size)

        while True:
            x_batch, y_batch = gen.next()
            yield x_batch, y_batch


class Network:
    WIDTH, HEIGHT = 224, 224
    CLASSES = 250

    def __init__(self, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph)

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.images = tf.placeholder(tf.uint8, [None, self.HEIGHT, self.WIDTH, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Create NASNet
            images = 2 * (tf.tile(tf.image.convert_image_dtype(self.images, tf.float32), [1, 1, 1, 3]) - 0.5)
            with tf.contrib.slim.arg_scope(nets.nasnet.nasnet.nasnet_mobile_arg_scope()):
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None, is_training=False)
            self.nasnet_saver = tf.train.Saver()

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            with tf.variable_scope('classify'):
                x = features
                x = tf.layers.dense(x, 2048, activation=tf.nn.relu)

                output = tf.layers.dense(x, self.CLASSES)

                self.predictions = tf.argmax(output, axis=1)
                self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output)

                global_step = tf.train.create_global_step()

            # classify = tf.global_variables(scope='classify')
            # with tf.control_dependencies(classify):
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classify')
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            self.training = optimizer.minimize(self.loss, global_step=global_step, var_list=train_vars)

            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_ops):
            #     optimizer = tf.train.AdamOptimizer(args.learning_rate)
            #
            #     # Apply gradient clipping
            #     gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            #     gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            #     self.training = optimizer.apply_gradients(zip(gradients, variables),
            #                                               global_step=global_step, name='training')

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", self.loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                self.given_loss = tf.placeholder(tf.float32, [], name=\"given_loss\")
                self.given_accuracy = tf.placeholder(tf.float32, [], name=\"given_accuracy\")
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", self.given_loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.given_accuracy)]

            # Construct the saver
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

            # Load NASNet
            self.nasnet_saver.restore(self.session, args.nasnet)

    def train_batch(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.images: images, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset_name, dataset, batch_size):
        loss, accuracy = 0, 0

        while not dataset.epoch_finished():
            batch_images, batch_labels = dataset.next_batch(batch_size)
            batch_loss, batch_accuracy = self.session.run(
                [self.loss, self.accuracy], {self.images: batch_images, self.labels: batch_labels, self.is_training: False})
            loss += batch_loss * len(batch_images) / len(dataset.images)
            accuracy += batch_accuracy * len(batch_images) / len(dataset.images)
        self.session.run(self.summaries[dataset_name], {self.given_loss: loss, self.given_accuracy: accuracy})

        return accuracy

    def predict(self, dataset, batch_size):
        labels = []
        while not dataset.epoch_finished():
            images, _ = dataset.next_batch(batch_size)
            labels.append(self.session.run(self.predictions, {self.images: images, self.is_training: False}))
        return np.concatenate(labels)

    def save(self, path):
        self.saver.save(self.session, path)

    def load(self, path):
        self.saver.restore(self.session, path)


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=128, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=200, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--learning_rate\", default=0.001)
    parser.add_argument(\"--shift_fraction\", default=0.0)
    parser.add_argument(\"--load\", action='store_true')
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value)
                  for key, value in sorted(vars(args).items()))).replace(\"/\", \"-\")
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    train = Dataset(\"nsketch/nsketch-train.npz\")
    dev = Dataset(\"nsketch/nsketch-dev.npz\", shuffle_batches=False)
    test = Dataset(\"nsketch/nsketch-test.npz\", shuffle_batches=False)

    # Construct the network
    network = Network()
    network.construct(args)

    if not args.load:
        best_accuracy = 0

        # Train
        for i in range(args.epochs):
            print('Epoch', i)

            with tqdm(total=len(train.images)) as pbar:
                batches = train.batches(args.batch_size, args.shift_fraction)
                steps_per_epoch = len(train.images)
                total = 0
                while total < steps_per_epoch:
                    images, labels = next(batches)
                    network.train_batch(images, labels)
                    pbar.update(len(images))
                    total += len(images)

            accuracy = network.evaluate(\"dev\", dev, args.batch_size)
            print('Val accuracy', accuracy)

            if accuracy > best_accuracy:
                print('^^ New best ^^')
                best_accuracy = accuracy
                network.save('nsketch/model')

    network.load('nsketch/model')
    accuracy = network.evaluate(\"dev\", dev, args.batch_size)
    print('Final accuracy', accuracy)

    # Predict test data
    with open(\"nsketch_transfer_test.txt\", \"w\") as test_file:
        labels = network.predict(test, args.batch_size)
        for label in labels:
            print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;2Wa~*j)e^jeSh{)<&u<X;JnJwG|@i7H9($f!3{DB&_<0e6kq@62rU9h;=$2$!#_S4>{R*;ABO<w(#-L{TS9k5_wU?Y)XU-7gvwqC>ueU%@b9&vxTo>o=X?dnHfL1c?AHsJGW_u!=b_>+DY%n`v$31eZd|=eNY`ZMA==l0x>c)x$}A}qu(rDKQF5t5u}*3SsF+mxG(qkY#f`O>&&a+C|=ABsC`{j(KQF8NN&;n9X;?v?_h^*ccSC^iJtQdPLVg^N|Ee!hYQZV4b7*3(MBc<XXP4@vHy!N2#;U9C=;)Nc;xDM-*~4ZYQtXeKZWW1i`Oy%IdOrDb$2mz(gv|?e<#5Pok=0J&UR3jB{oy`bJk9L{%mX?TrDE&+KPXG#`)CVk`JPOJ9>aA#!KJI#kg_RQwt(}z6_5k;uO+uC+wqljR!{q?MTv4UgeJ#^E%wSWOz=(kVX*OqLE$U)m6qyK>-&aCA6;QW$%y{qs9UE&fIf?e0PQE3>4I~*VdNAfrC#vv#~`m0Yr1Mj;e)G$^QoyzcktwyX5$cWEsoZ>%~%}$&ndxCp;<i{Tl-k@6H{Zu*3DtUTH%Akxhddr&y?~Q(+T-;)4V*_LDDN`CM##B8v_XU5l7)`qN8w9Eh5j;+8&zj6GxMfgh0u3dOzNbq)b{9}bZcpY0iFnz6R%^KMW{PL$*v%*p-Mv)?KqZ^d-?IdaXJw+02hCK^@sW@yy56@z9Y;?C@4tv7>xIkTpUgpw$|briXHc=rnL)D&17?Z1s3FgJg~6h&G2W{2s%6Q3TQwFkaAuhvcoWegh%5k_kEOwb!3C;ZlMuCAth9X+`7*!Y9!4(sMI$%s1;5u%9Px`NsVnd~^oo0<HEs?VA9zKG(TABp6e`RDSdS>HN{Gqh+8gUIiHGJo$oNReed$Zt{OJ3CO|(O~V{2-FRp)6+q(KT`kMtX?5Lni=o>I4u5{H?2c<S@0?yLW%g|7NPn=*J2~NuVQ-fViJ0z$0SC<9&o7%yOCr?z@Sw!kFOCo7`UYXm-4x@u!1>&vAWa+zaY?<`>e?SD8u(O4DLf3?A85g8rz3vA~0%9Zv={i`f$*)zDqlU%CV0KEvysG2(Gunox9#0mt0zUMfEz^dj<#xXrN~^ojhYHDPZ{Qom`+3o#(RLLPw3XHq-QOI{u0ElEtjtcX*FF-B(1a-m`7$K7ic)M|r$6q@m8i(p6!LsctrW+R@?z*D(8%QIO;xMh@3CZV-fK$%)y>7dTRvNKbznls&1V2yt;Fo!HW^?zd%?mUX_GHk<vo8@l5F|48}45-^Wd&4Ua$2ps#lxr_x}uEF9M<%_=+Y^sB|PN>giV_l-;?h0qRCVNI`YkST0aYIgqs<glDSII?Y?#cnLx%#;@Rj7!37qGwl?B0vZv6BwWj2E>YYcFE6%z>^bKFYZXBaJsWP(VVY$K!7hfr<&>wtObSiSM4iR}ECEQzQWRL!+76i^9g)Z|jAyGhe;(vC?@F^_g(AL97?yFR(|VLxsS?hU0oKRTy>H_J}diH6TRF$SWDBlFF8qrgpf4PA^PyIb0`;&yge-TF;?->DA0hhQd}tU=d(V5HhOg{7w%7sIz_l&et$+a+{I46c8WQ2n!#>n9w>6P|!}#Yi8t<R-izZLD4vT8}(v4YaU$83un7g>dmM`&4}^$X5fc^J||xpGn-WT9>N{QWNfh#Uw~HeDBx{fk`UgS<9?&KGboo|DHg^1R{>UoaWG>Tz2n|Yq05H~)Q888!2@H;%0H3GqN(72A3P>0>xek|oNzVBW^S^DA?d#bLoh(ZHg}Y5yJP%ryxs~VEls@;#SqQd0)&P6zQuK3T40B|cg43XrW(5rW$g;(|7=72^<mMLA5S}=J$xJdlPj6LQZh@)ws!drIi}V3n02ZoE%eq?!NQjvAX;WO4(iF_C{_0*-v3*@p+dE0Ptxj487k7zxYn9QaJo!q1*2)+U{X$a$5j5FM}R&Z8OKX2>~Jz1#;W*~y*e{6$EA+@Aez*;uX(?bfOP*o_OO?P98s40(WgW&^RW73{jwK3H(GhNBc?PR8wYU-^Y_sq|IW$0e>hKK9@qGYFAZLJosm;Y*!<R&e!{HLDnWbuZaCF6(^=>;5qPU{7K+LqZINoYJpXdk_V+R~%E)ZV$h0gP)Q<5Tr|UCvn2<fHhuPC&V~oz1p5VtjVil;(3k<2jKWB8%TPaZ*;ElQOp=*bFWnA9^AE~LrggK(JsK9)hvn)tI3P<wVXnjo4<>2sCw`lM=1K2zdUk3T(py(}%t<-ww$@PWS%M46L#2hmmBCK{3!7LhV;(afttf$G-<}4`b5ym#yrQ$h%9p@?C&_wU&;F{ypJ@aCfUwxTNWPF`G?oM30_WqV^7%`vqZf_)lmQX6IirkmNd?BoC;r;|Zv8J?2yG*wG7PNENpzd*%=O{pv3|2%5?nDgP+&dB~^<#0_eD)}pKF%bh)}&G2-Www)jO){bxdf_S7&^30=DqRki-C_rJ&lwvwYY>0)a0zzgjyLR@?FqJaD!w4IeuWhj;Oy9;2zhepbY=BsBMRJgb(-NWR*Sir(Q5{^T<dP6Sm^t#jQh8)qP>q8q7vd{xFlZ+*@>H%eX<!$Eqb%^vy+bSASeT!QYA|3sqLOPL>w97-puIE|sg@22#+WfLpa3x#LV1M#;O1Gguqp5K`Slw6e^d>9l5(vK{;v;Dl>YH@-&Npff!8{7X++QE`tY2<vNknb$q?^e1*za+Us#?Mb;4M1spa{iaKN*JWR&+rQlwfxJB6opUb5r(IA{qpv<Uq9Wr}=>eb0KL%s|ZRn_p{vgZqG;pt#Nuc)Itn?*BBI=Fc=coj(3dhgJxIY;~yk6q#!E*INz%gblRH6w{3JAYSxI8SKvpwzB_XNsrz+8~Y?;35cPU{q3hj)2lGkeGF>42DuBqHG^v6~-!!d2y{g<i|rd+_k#nqaE!t5-VH9AULvDS8NdTZm{NB!h36g<NO>v=?WM>4v26kf}6Zwnk>zXLGUIR#6nG?VNtde|qhw@1j^+jewdB#vMi2E*m>V>(2}c@gzX9d0RbZBwofh2uhY%K90+d!*Rt00+Rm!Er69g2Ov(Cgi0Uo3&3s#>O*0zQebTI$U&U|Y_b(*H#=RnkC^>Ta;m~>cmcu_DEgDb5i!WI5g%g$)O9?jJ&x~u)ncMvU_9i_Pv;p@hF3YOgQ8M1LG`lk)(8xG`F#H35pSMIx-9nsPtU|L0TqY^bMzIVXvjvP8xisEq&#{jiUylzKcU1^Sdc17C=;aRav8{Nv4R0?BN~IGeQF|+7!$9vV=j0m?OSmzt@>*GW78;tLRJV|BkG9ZJSuUYe`z;18XG1x6~Yk~&E{MU`K3U}eI~F;z71C-HBIqT{(;hpTb%-*Y8~s0dr!uMuwlURQb!iZ-C@d*so=xxt~^CJ%eP}W(;r%1y_hgUc3VD&1;^GN;XtKk%)e#<D!?s{tBgvOtZvs-9J<4_6z_!OS4+$t)cH*7SVh?GH0|57x%v}0%LzP7iChh?l2eY;f`q=N@`YI%#(Jack{FkEFVndNe8Q$sfu8RGguIvr?P@eZ00HzBq&ENnXX<+fvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
