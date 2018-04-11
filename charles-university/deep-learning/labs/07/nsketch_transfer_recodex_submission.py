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
                features, _ = nets.nasnet.nasnet.build_nasnet_mobile(images, num_classes=None,
                                                                     is_training=True)
            self.nasnet_saver = tf.train.Saver()

            # Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `self.loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`

            with tf.variable_scope('classify'):
                x = features

                x = tf.layers.dense(x, 1024, activation=tf.nn.swish)

                output = tf.layers.dense(x, self.CLASSES)

                self.predictions = tf.argmax(output, axis=1)
                self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output)

            classify_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classify')
            finetune_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            global_step = tf.train.create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(args.learning_rate)

                # Apply gradient clipping
                gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list=classify_vars))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_classify = optimizer.apply_gradients(zip(gradients, variables),
                                                                global_step=global_step)

                gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list=finetune_vars))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_finetune = optimizer.apply_gradients(zip(gradients, variables),
                                                                global_step=global_step)

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

    def train_batch(self, images, labels, finetune=False):
        if finetune:
            self.session.run([self.train_finetune, self.summaries[\"train\"]],
                             {self.images: images, self.labels: labels, self.is_training: True})
        else:
            self.session.run([self.train_classify, self.summaries[\"train\"]],
                             {self.images: images, self.labels: labels, self.is_training: True})

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
    parser.add_argument(\"--batch_size\", default=64, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=200, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--nasnet\", default=\"nets/nasnet/model.ckpt\", type=str, help=\"NASNet checkpoint path.\")
    parser.add_argument(\"--learning_rate\", default=0.001)
    parser.add_argument(\"--shift_fraction\", default=0.1)
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

            finetune = i > 4

            with tqdm(total=len(train.images)) as pbar:
                batches = train.batches(args.batch_size, args.shift_fraction)
                steps_per_epoch = len(train.images)
                total = 0
                while total < steps_per_epoch:
                    images, labels = next(batches)
                    network.train_batch(images, labels, finetune=finetune)
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

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;2Z@C{apYUjRN^fX--d!I+rlqYfQ_7t}hB50>>K+=LH9y>4SDPC6;W|!F4QOtFq=hF1}0EP#5zc+^O<YIwt$^Bur>uQx;yQXLGI=Yvw^xV%U8Q0fL8WFZEd}ufo%LLl7oJ!b;AiICyNbPET=HIj<zLOI<8PP4jJYA83L=^^b*T4*}Z(9>jIe-P;8DqX3Y_G5x8|Zg~p65LBMb9K@z)4;EUP&JmyTu!UXmenZmz<)T0~<NG(wA<4~0a0{J!gh`QQcu|{%z1|?8V*@pmk+{^madVS9v1Fy=(8Qi3G;`+!GP>_nnAZ%WC@(>@!NOQ8=l*llQ6m#_X*mIiok+<^_8%1S5o?I%KZi1cR1k;k=7PlPCJDqAoXpwGS-$R=P9b%nH6ks}Yr%#YP`7n?Xj+axow+6$wd@u8>cDaV64NXh@?uybAgsR(XI~5sk8|y|AlNYhMj!^RS0*A~YtS@5xx%ez9}<@zkJl?_OS!qbgOYuQ`+UCr8TFiBl9g2k@1<9f&$P;eYo6V5BELC66EX|PVUQ29x(z)-a;jAwb>iNjNieEv86cdO>8=&U5FpJ8SBf=b!|Jwc0f9{`Zn-$NI-J{x&lB^))rGMjeGD7TlQ7G+Np_Z2VAO%0du^8iq17=o8Z1~_(o>Z3<|piGS|JrEGQVDsS`&fb*QO=uZN1{>BX&Y<8Jx@ENl1rR(&9cx{wIzSh3V=*WuqScEQ!Q{7~b_XrIiKh1%h4OA$=EWcLF7JAgNq)U<01DMhI2O|9$czSrJvCRX<rFWY(uOQuFBuXWQd_ow00(;7Zi8iOqvkOJTa-|0lR?z|Pe9=sX1Ugtlla3pQCHv9V;*ct_vYj<Un}UMDGTZzz-nk|n8!B5<_Oa}QO#b>7yM7tj2Y=1I^_4CIb(sTb$W`^DKS5$mI~a1~gEhOI+-dRW7D^pe!-c(zn<r~2=%aGEZ%uKMrEpO<|8qQ&s){-q;1x@R*@LNCMxM!a7*&oHU_=rp7E>OqljyJs59MJ`p`??*V#c}y}}Kk(ugMZQoiqvr+31B+9(Y1))oSFSerzYvWlpgcu0jPdYMP$c56)Ocg%5ied=oM`?q1v&WR)0u|?JN~w)Lkfm)g*oPig#!zIrxNEIfNU}&-(cNzf$M#$)+IVQ`hst542>g`+(5aHGv$Smc9>saR*{s+^yry^$$*VcXVecny4&E)?cLsk7>l}7j{r(aNYzc+s|*(XQXw+|(yH}JxkOS#&!RW8;Od(@JU(Fzm|Z9?c>PEI5*ONK1w)|q3PuuVGa7wCf?SpxD^55|vN0=2V0|*5&<7tO<5&S6B_uAlacCBj9)5BdETPu4v=BOoseM^U&E>Exajl5kz?9JA5dfupH3Wu4IbK-bLVvz@H!IWkRRpgwGB*<nPO-W;{VHiH0uj9KFulhp<6x0(srAK}EHKTK7WlZzpGU|zxW@Nd%=zb>iNY*bIF(r?!KPP5xU3g-G;^+r`jgJplO+Fj6}$FMixJ-_3Mi9M`T^8{)+xFyq&8)kpXna*hUnDw{8q-e^7Fh>)pyo@U%NHleFR(sSUJDO4btkwAh{iD53l>1{#_Z+sibjliz#ySqPEAxS&7mviXP_JGt?Sb=1T`jF7#5$1Ax^}iI^2wFDcdCFz6$O^6Y-7Kf$UaDuZct4AQ^-AY?D=eD{qK4<*Zd8C<s;Cd^`;Og2Ayut@eUE0qZwPB{s6F-bOQrphSA13%lNO1itu3g2UUZ-ij>OP|A#OYZa56}Xd!o32r{t8~;!48#GL;t{x;f>rB8hCS+LtEK$WFByB-VY+!4LRA?1{#@2v<rrLtJyR)jil0+`(oYyg_k~LCFWPe`9mQhsN`Zwvc@%l8Rrh-#h!L_1eK#iW{eG;<WC>;M%y(7{+Sdmk_oi@RI;28!!Tc4^4<^AE>X`Snw*m#tr%<~X!375v!BU;gM+6`%WCQLc)Wx2$?SJ^>YF5s_k%3<Fu3j|n$n=yr@jAR}^nP9qkmgRGQ!ame%9~)eFZf!GE`%QN>x-d0j-AuzPn1NI_ZW6Y%62iia5ub0h`b2UyjshuY#l6iOn+}uUBINzkJxX*FKdY2(pD0pgS>-<d~I|}Fh0Z1+?W)q&Nv1%G%DW+Bwbt7&*`(1x6s(TgF1Qhj(|o^OYE4S_NMLQ<6_Aw6vg+&Uj3kdqnSnX0rFUFkkGtW?GQ(+!Q?5zowr_SY#t~BL#{VhCaw9nO&M8ZHcaJ<AJ@{umh>+HZ;2!Q*>58Vq{LC+3KGr|g9@xSq+lEfz3-Qmh<_sifKOpkvF%+$m;cz0RO4O$7!&rgGtXf3YI3MLSGb0b{<=nA;wi+c3%LqUUE<?ErsmQrbJZubKT|O#Fb7-0nIrpi-XXDHB2*Q>R`Q0<0*@A=*l5WKwnubQnoYQ!J_>~EPAz0CG3boC<&P31;%lZ+UBjwDnsU2lQXvsIH}h%o?fR)~<|W_L*ot3LY5DE*6&+SDJd0j=E)eU@!_-zx>0p?LUI`V2QTBNf6bR4k?(7edU&b++$I)s4BJ3}}PBTb(4koODbfnU#V<?C>uNdxUzT7R8W3qSQarxd-8?kbh95<`R#fx4ed{>zS9Ls6{R51ho7vqIc-6@*z%2hNFtQMCF!jCVI=?DNVVEzn_vEJ+VPU2aMq)(i@&-vt6G>D_|UF^k-q3D4R`aUy=V_n^*;>#DK{lck)ATE6U$QG40->mFDh{$pDlR>n(<X8Q;(ESC%{NR5?g|dyWvCc2=K50d|L8cu7+4h_=SAMrL3nq??RFI|+@4BcoXRAj}cs|Zz(FyTFb;L`<Ql1Sc*0brVx&IaIMOL0ZBjyNA`fLf``_x?l{3*zK4PC>1=HNS#pEQnT6(V8&WS+nJXEJ4JQ4uUm0z3pJ2`(Wk4h0nZ2>=PZ*ryopjW5i%#N>WlJwX#4lm>OsNUo@@49M%Ty+(<aZ%kayf;RlqabKV8%hXF+velU7Qt+IYkc$Se$9F`XepT^ps$exC0zem78(dXB9jhGo&~I2W+|H6wl$<S-P#Tf1dPeDNB6d5bL-J+Io?yM$_Q>(Zet+R&*v<h-Rth726EBA4`3Z~n$dn6}&U75`b03fX3<yFoO+58bY0m=6Rz85MGF|ycf!r?hj`1j>a@I;+ip%9y!fu%^e5^D{w5dVC5+RL?Mvr6!c=uERQ67Mr%D7Lp*LRSrRv~wiPIX%bxW@B72T|<?&Kledv8{0o@W^rMcjm0p(;K?a4KHj1XzbX;-<E708h!7o%K}!X9x2=KODfPI>}_akJ(z3IQ~#(=U+Gg*sL_VWl$%YyeR)+AQiLqwgBzBi!6bZ(j3AoYU{8Yetx&p~j}oAe#0wV>@IHDf5=uXK=|%jNbMPm7ZD=avPEgX~6sW&<8Zd?pq*DMCt5J(Yx?q7OeV~4oy8gmiAYDaZcw}_nJu*0}5)0pG*Ich^fWnntMHrZb$cJuuI{tb)3|<>W?_*5Twu7k`FX{Ox0k9xc4>dBS$+cX^UPWr9G*47{=v^C*-XY6Lr^G>49Lr=xoG0MfKANl4C*XuRGnOsu<!Q&4-}Ls=7$mRk$}n|;sX-8=S4(35EmG1ex=fw5&jE78n$TOWxUC5H20Yn500000y`s%+<Y>FL00Eg6hByEKyR6pqvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
