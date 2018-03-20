# coding=utf-8

source_1 = """#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials import mnist

from capsnet import CapsNet


def load_mnist():
    gan_data = mnist.input_data.read_data_sets('mnist-gan', reshape=False, seed=42)

    x_train = gan_data.train.images
    y_train = tf.keras.utils.to_categorical(gan_data.train.labels)

    x_val = gan_data.validation.images
    y_val = tf.keras.utils.to_categorical(gan_data.validation.labels)

    x_test = gan_data.test.images

    return (x_train, y_train), (x_val, y_val), x_test


if __name__ == \"__main__\":
    import argparse

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--learn_rate', default=0.001, type=float)
    parser.add_argument('--learn_rate_decay', default=0.9, type=float)
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    (x_train, y_train), (x_val, y_val), x_test = load_mnist()

    model = CapsNet(x_train.shape[1:], 10, args.load)

    if not args.load:
        model.train(data=((x_train, y_train), (x_val, y_val)), args=args)

    accuracy = model.evaluate(data=(x_val, y_val))
    predictions = model.predict(x_test)

    print(accuracy, '\\n')

    for label in predictions:
        print(label)
"""

source_2 = """import numpy as np
import tensorflow as tf


class CapsNet:
    \"\"\"Implementation of CapsNet from https://arxiv.org/pdf/1710.09829.pdf\"\"\"

    def __init__(self, input_shape, n_class, load_weights=False):
        self.model_filename = 'capsnet.h5'
        self.construct(input_shape, n_class)
        self.train_model.summary()

        if load_weights:
            self.train_model.load_weights(self.model_filename)

    def construct(self, input_shape, classes):
        x = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv2D(256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)
        primary_caps = tf.keras.layers.Conv2D(8 * 32, kernel_size=9, strides=2, padding='valid')(conv1)
        primary_caps = tf.keras.layers.Reshape([-1, 8])(primary_caps)
        primary_caps = tf.keras.layers.Lambda(squash)(primary_caps)

        caps = CapsuleLayer(classes, 16)(primary_caps)

        output_caps = tf.keras.layers.Lambda(
            lambda inputs: tf.sqrt(tf.reduce_sum(tf.square(inputs), -1)),
            name='capsnet')(caps)

        y = tf.keras.layers.Input((classes,))
        masked_train = Mask()([caps, y])
        masked = Mask()(caps)

        decoder = tf.keras.models.Sequential()
        decoder.add(tf.keras.layers.Dense(512, activation='relu', input_dim=16 * classes))
        decoder.add(tf.keras.layers.Dense(1024, activation='relu'))
        decoder.add(tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'))
        decoder.add(tf.keras.layers.Reshape(target_shape=input_shape, name='reconstruction'))

        train_model = tf.keras.models.Model([x, y], [output_caps, decoder(masked_train)])
        eval_model = tf.keras.models.Model(x, [output_caps, decoder(masked)])

        self.train_model, self.eval_model = train_model, eval_model

    def train(self, data, args):
        model = self.train_model

        (x_train, y_train), (x_test, y_test) = data

        tb = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                                            batch_size=args.batch_size,
                                            histogram_freq=1)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.model_filename, monitor='val_capsnet_acc',
                                                        save_best_only=True, save_weights_only=True, verbose=1)
        learn_rate_decay = tf.keras.callbacks.LearningRateScheduler(
            schedule=lambda epoch: args.learn_rate * (args.learn_rate_decay ** epoch))

        margin_loss = lambda true, pred: tf.reduce_mean(tf.reduce_sum(
            true * tf.square(tf.maximum(0., 0.9 - pred)) +
            0.5 * (1 - true) * tf.square(tf.maximum(0., pred - 0.1)), 1))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.learn_rate),
                      loss=[margin_loss, 'mse'],
                      loss_weights=[1., 0.392],
                      metrics={'capsnet': 'accuracy'})

        model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, 0.1),
                            steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                            epochs=args.epochs,
                            validation_data=[[x_test, y_test], [y_test, x_test]],
                            callbacks=[tb, checkpoint, learn_rate_decay])

        model.save_weights(self.model_filename)

        return model

    def evaluate(self, data):
        x_test, y_test = data
        y_pred = self.predict(x_test)

        accuracy = np.sum(y_pred == np.argmax(y_test, 1)) / y_test.shape[0]
        return accuracy

    def predict(self, x_test):
        model = self.eval_model

        y_pred, _ = model.predict(x_test)
        return np.argmax(y_pred, 1)


class CapsuleLayer(tf.keras.layers.Layer):
    \"\"\"Capsule layer with vector outputs (instead of traditional scalar values)\"\"\"
    def __init__(self, num_capsule, dim_capsule,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)

        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = 3
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer, name='W')

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_hat = tf.map_fn(lambda x:  tf.keras.backend.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=1)
            outputs = squash(tf.keras.backend.batch_dot(c, inputs_hat, [2, 2]))

            if i < self.routings - 1:
                b += tf.keras.backend.batch_dot(outputs, inputs_hat, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


class Mask(tf.keras.layers.Layer):
    \"\"\"Mask the vectors so that the true label wins out\"\"\"
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            inputs, mask = inputs
        else:
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.one_hot(tf.argmax(x, 1), x.get_shape().as_list()[1], axis=-1)

        masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    \"\"\"Non-linear squash function acting on vectors\"\"\"
    epsilon = 1e-7
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + epsilon)
    return scale * vectors


def train_generator(x, y, batch_size, shift_fraction=0.):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=shift_fraction, height_shift_range=shift_fraction)
    generator = train_datagen.flow(x, y, batch_size=batch_size)
    while True:
        x_batch, y_batch = generator.next()
        yield ([x_batch, y_batch], [y_batch, x_batch])
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;7%SBT3rAc0)oax6*GjyvYTC`s}Z?E4>dgkac{%n=7fv}Z9C2yLvSj;GvI6~#Qz^B=K$n%0+Z_d+C_KFzILwpcdH>kw|Ezv+<XwK;+u~kCIG8b*anJS!qYZ}GW=YDQj@#}md3BWGf%*tbs;|bI=GjWo98HwhN7~}kJjB{<Rzv?<Vpw%MHO*J*QxzWYy<X^((dmK|L7EEowmLX#hse}m792m+>irV^s=Ocf>{-}45&PRLS7vx^({2jxYfLdE($it*KD%d2J%|#1m6*}XhG04>{Uu#<I+pj3>3Gt?^<Z`1Z0r4Zu6?E)u-VGI@!iYuk|TJ4EL=$ToC>@<Dx+YojIPz2^;TQC8pPLw?Q<n_^Y;V1o)+Wcbc<S9iMR!zyqwE-|Ac?%^oX(AZ75&FV*{QL<AIM3UV^62%POqHn8n^)&xb2zK=<|S@7b<6^`pq!zWBUG*H_i9m{P4l?9RFU5(jhEtM1Au^B>d$jyEO%`9F~03>}ly)yvkwXscN-<SjdNtBTbtWh=t8R|<Ns<Xo7<ETFQ*9mVFBPehp`Ao>q38UyDRx&{vuOSCsP{AsMW9`a&KZqtA?4o0KXFOqpv-*Wluj~nBOh3l-AzChqZMf@-^BcVqf^lN3#AA4|^_dzmPSL{}tWE28-)U7mz@J)A%VsEBaTH&V8cMeAHCB(GVqMrA(5eWrEU}_B?Rv{ly|*!nlT{g8ozEDbX?QC#C7~u(>V|{Nb&iAq5waf?p3k`EeslYZ1$Mf7$V5gM4Uy?4{K!!Q@_@Iw4P+m(&RA<~FGt>sC1voJd0Ots-8irq^puO@NrhrO)g-qi#RTdyO4FlGMLN8=TSXvgTa3qI%|;6qH6Eu+VJ1fVxI-49QSo;?DaCK+^&@h3IV+CwA67w3ARJ|WGX%qI1n}cYv6n+&+S*02u${AB#2j$WQF1tyW1MUAMCqWbrw7>tVya1g^AubaYN(p`IOfG2MZ+-ScW?y3fMz3pSicD=6iJh|#>;<JfeHeIK<_<qT~7N&4)-xu*LIX&f<2{BN7SHIF#*mMoi<kv*Nv&Zmi4h(!@F<<Y))_DTnX;;SHdv>N0zF<B`Y0d3*cfm|8S`$btwuS-`JRkh`AL$T#}M;?<7MFc)N(6aVe9icCq{;(P4fgLyEpIu1e6@r9vJJp?=|;%@Upe&E&SN#`8M}?0Paa@u^|`$A#HQT%OBy!krw-C&U^G&H!8oxVmqMPh=QanwXkVQL)i*)I3U4DI5dk(zpRrl(mMY#TS<o4870_+O2jLNDyr3)4L7#;xI#C7kt7g!S#&oWDTg-EO2l;bl_lHnJNUnzbB-0aUC;4ayD9>J+r`$gsD-?Ke?2bQ&O3lEcw$KSX%r=H<9NG=4wICas;L=m+-MrGhzezvL}?hoySE<G2kSqc+Twz+`G%FK`>ybhD{HQ15VzT8(6FJUt}cSeEneYZl)tnh!I3G><T@Fx#`Oz>M}QzTwpj*qk3r~s66nigyo99B=Ku1!6?e}ceDECS=PC_r_DngAD24(V`#Q2^wJYJ<?Y;We4@;BXCkU;eGPi_1}<S?P=W>b4M4es#BI~y<YB)E3zkrV<Jb)*+KASmfDR%b%_0k4K5q37cLmhlilmRFCYH8p^LgDyXQ%hRc|#99%<g-H^T*iGWT;Z^_Meo)$JueQ2tutz31b`V^GrA`orufRfol#ALAoF0ma^Fn08aJkn9oOm8|?r4$abl#Pos)0IxJ2g%BJ9uTHJh<8+Lt=>0g#XHo&}zy#^8_WqFxOTq-HaKMl|0(15<x&{|DhiSrjs@B+`RvFDOsvTtzEeD7VN@h~(Gz^b%Y9k;cy>Zl63O42*^>;H$QI%CDYsh2OwCG|Pr<7!l#Debr?SYPp&)w0)7MD9eyPlOjw2E}jozf0yasqiqiqQ*+R$e=$BVS=ka?E%h9O5}zdt?=fY^xipq6j!H6rg65bQ_-T(GJ1pHpYlJO8L*{_#VcJtyi(DvL*u>J39l37o6!#%xt<_z3AtbqqIL0=`?w34s4^x6Hh6c*SN-(Xm#zv~$$7bdS_DM+@-4c+6*y>f_2LH{OE57x#BEF#VueKFi6d(caRHxnp$z6<Gg1~S@7KQ*hAR@kgumRA44e+v6<zvTb3pchAB19}zfu4RSUF)hLW$fSx%lA=_PKXLk#I~9iuzo@6174fV9U+)_6<XJi}KQO#tZ!T_MZoxXLE<V3JP_&80iI$VXF2&ZH0wcx@WE6pB1OKE#{<#qn5*6-EcxAgC>b;MOyLkBhgUN$#BG6kUm0n?m|Ik*&~zb;|G$QR;=Bqj5aopZ1sLky`S=&(C0m2-pH-8_-})QaJC&`MnP=@U<B7c`*rwWTk>|f)@wBNQq{>)rG$UDZ54LOffDNrZ8o~#*~88?^}!H4xqbGO!^=Q}4B7yN67TtFB@4uDHJmk0;|TLjet8;tAH;#^#K?KnTb<Yat<)miEEctzpFG&zckXrI5sCH%pby>YXg7s6xx7Im6yiUba6TBwc-67JKUFdmU_rTuMib!L*J4EMD{Wwl-JKPNtq2W-@ARH`Z~y1$+oC0aa3(2RAqLHbd{cOlo3N+4NDht*kl>VsMhmBKD|Bo|6>L+iCO>Vz>j9s?T#xrMYj!vRFlx><Vj5!?u^LfJtd>RF5mC&|H<#RBS~Mpqko>OkQXLa4hXMhSY?4MtVZ%xMH!-!dF@_&2%}YwEyt3YL0?i)G_v4+!lmmE|4VFhxz8qQOr>#_+I`Nb!YDEnl44$hQaR;|NbxLINZM^j;L;{~?#YeMVkfZbc^YVJAsPGf$kV8RgiT3~1>jC&=b;(FBV%7O7wE*-WJPeRf>g9%{?z*KWnu8!T;qY^toLgE1HzUnmm0pm4)}rBlni&9qsEEe5S>7$QBmd^*=5(>K19rYr$L2b|<Dbxx9vLt!I)<!!>A6DZOn~9piK#_JvUMWfr_YN%qebKS1ynK*YC|z})C2qij37dJJOBdMQe5`@{qLPT_P$eMc5L<h!~z?oZf2aONA@BSwIZQ*@!DF8xHO5bABQ^@tA}-SnG>F$(kcZX@6f}CR0_4kvdU*6-73TaWl67d=)Azp5V=>XJxn0)+V)FNl$?IJ1-0hQ>nYrrMMzBU2=Hc=%T6RTAkc<E7pDV<z38TB)tyIRBN84sXu^<I@ko^a+yYKn26g9&2Gs-gQGpV*tvPVrkHR!Rsl*)q*o)qBW)%Cw`7NAk8Be((|E|7Jg+$!9-T@4WW3HiGM^Uw6m_}-sXq_=ow5PqkHgPmdEtE4r=I#+a;%+@8LbsV0ju?s@U@>G=^b7Mjh;{i_0P2Bz&Zp<{Dc=1?Lb||N0(}Cbv9Xr{Z1`*PQmSST$--9oj^kQaVqQNeg+H^qFTnu-hs$Z0aBu0fM-$r!RMXhcKe{$r_`A{K6jZltRbhqU;O%&SPoDB2`;!T;8FsN~H;)|oH}}pa{kdJCPr!Slo#lmua=^7NgTg5DKBN4E;6R<Gz@Is{q09#I<{V?|R+ZcIMewj+33Y3_plj-)#g8{j<rdLnSoRv4$-))#1qU8%h4O&ZX9Gmy!m^654*(T#VmW+*iF)-vB{73z)H+GjH|7vb7;aOAaVPD|eSz72?7QUQePjwp00~G7_p;Ns`rV<9v!c<q#SK7@vpxJYC=fg8WC5G@lH~J%H-c6X`-f)|g=uzNzCnH_+U=H@*QF*noa`7ofq0j63@#CQh(wI&welUNyz-gJF>HjfcGhXX^@gcgarjM@y$=pBu~nOvD0pCjr--K17r`TU@Zm^c-qHbKan`*kcZ!Wa_3Sr1!Hg)f#up)%T$uYmLu{w5ow1jH#U#gOPM~57<F<~b^675feNtv{yo|@qzkV|ECMQ3#LNIBIbs?;nuk5AVStlAd2_#KVIbpv}S(^;#T-x*m5>duVc8~AhE-}X1utC#bfAlY&9SPmGAouH40ZT`ez##(+aX*$<t0^YJ^&P$V2LJF6-9T{fMj(^RQw>U`37G4~2kvDy?Z99p)VH;yCO^=rJl2K57rLG`@Ag(@P(qVusz~F?O)@^|TLZ#y>P>u(DRK`~Re8$N@*fnE^O@qwg2Npgr7D3Ewvun#+AFGmW(bjP3eGrCBJi|JC*kLH3(znVs0hFH2bCg1tKiZ?u!R9sY?{lFX7KaH!9hDgo}T_){^OGAo~>QsH>xdq>Pucz1VJ!Wo{h!NV&|<_S;5|G602@mSr`_WwjeEU(Dhkh9Pjk`*Y;v4=p%W>9PmqfP{OSh2M`?F!f#=k)43WZY#jIX3aHzFo&gnVpxCV0$c~l4s3E4QwthHsl@L;RaBC#QBR2GHM`4)E`15elFjPUAI%Yd7b}FOP&dTfwc12b#%%CEmxN$IJ0M~MW%&iM$u}L=~n;iq~z}o3LA=)1;MjpK|Z;TUtRo}dS_D7;Uk_st8j{ES|4Ct|u@L>viHuq$CgdK$50Bu&AZ*eg^k+`zAUbV%BGlTW~_N;~t<cb~nRzR00Z{Kmu6I;vpEF?EY!R?GZb@`(yWU(zw(wAInDmZbyT>UZAos^%P>0C3sg6iUo%@i@ydL?Bm!C|=GRT8N%OhRLfWE<V#8j}hb@2HDZ1TSWRD=~1K$4iR2SMk&(cu`d&rCyjtth6Gn)OVJ+rXhl0E?EVaMPKyHl(Vt<l<wM9AFM>3Q#>2TswxH>rLX}BOUoRBiWxvlj@9ZmWi_wX#iZ}2(0`>X3qh|Y0S;)+EMF63^LL&OEty2&z_Q<*#g1vSsD#AJVcV*`T>f3iHTLMWCW5u=h%(?{10o?Cz-^-<{2P)$-DqC!cL)!F*bP+KyzyAB1rHcizJ#&Tg-l|dA3tssdcXn_S6~uf0~9O<i~O^v7Xobyim0Hwrbmf?-q0!R@UAgbTevgl_}%aO(jCbgrLe$GfE^4CU_BY-X3%#V`Y@l#Db_77Efw)s3LlOOpiOrQu@u!NNOm-k%5<;5?~CBgNN63uDB$I#e?}R%uc)ap$B@$lA!<X$sKE7HbA>~SBsl5{dEGwDGnO<n3O>oH6uyP0w_}n2LlJ}<sv*%E7N37Cxwcjnm~nzY>{c1`%BC3agW#FL_<vWYd(nm*Ug(EJQ}H5fIY86&z6k2Zn76sw8l@88QQF8KR4<KeaAkq3Kvq|TGD3ADVt^EYVI_18MuX{SwNjv#zk34EjZQNa5!a`0zU05dX~rnO#K)!~hg-CEnAu&_MHJ529DkC2{QB2`b(;DXC%Fd9-O3*?!@Pb~d-O}oZM+2J>cB{GObP8YLUtD-aWTu^Fc=ZKfcJN)FDvxx%$J4tZWgh0<O^u>f{$@X_uuVd9EhVi7U8X*56fUkHlPn|Wru=$4X0KHMY*i_z&^3dOK)F0ONCi*<G5@tcPfj1as<F^fMhQqxW9(W_Gt5?N)7v9c%hTDrtX}666I$nm6=Q7=H-?!)C`Ebj~0w9KDg&zdqXEJzlIb8O+s($-n_Af*kA_76kzRx2i!&_-5XB;)I>zQ4+~7PE8lLXz*aU()!ROYUu6AH?(oj?h#%u^7RKmNDWyoC@TSYT=d-4zH#KA!emps*`6Awd9k3z>&4n@|3avp1CUco8|K@2-`%ZM}G}SS~Q_NM~sBBwl-TYFHDxYknOn(YM+4Hr<*lm2yUmSh%HgFi@kVk8?M-eav*-#If8yqxK6xR8p5JGH+u~97m@tYGZ_fn?OcBhVLY3VNFhOp&FRM=%u8<9G{{r-@Fq-hrYM$6F6jo3d~W7fk8--Qnt8awsn1^j{7#@k|-Pv7%x!S2cSXqN{gCEfOjFjQUXt&?&f$}@TXID#(AMvE-~NJn@^+$PTC45G%_AD<eC5D3yu5iBDe4*tDw|Mj@K*fBXY59-Z`e7-JYK}W=rlfF#D>rWO+k~G|GfR*-dJXYXS+p5O_8!6Ut@21w@wmupRZtYL^GTyy4<cV%mgJ?Q*ckApKrZd8HlIA`U5lmcACT{oj_^BUHF6puQwlXK~kM&8==WAlZh=Q^?DJ#%GIkhzI4JOMC#Q-}OKZpYrnX{~0RdYN;kL}t}=N4dYN<P(3j3r?ik>R=TrH`WJTUAd#S+%o=#*S_3LL=#XyUN*Iza(_>jj!}HOW;{;B;^_)&1+Rz>!J@~RDU}^tmM>XP2+qwBoK{GP_YJW<&eE(rp>`2#$zMf+yQcV77C<3RjgnTzU)L%5T<HEsIW+YI`C}WhHwJMu<lwjs6hPgCqn^C66OFjUrWF7ReH3eI2FTLG(#u8+2cIhN5Y&>nI%Fr7Y0GsCOi1?yjdH?!cAmWZqHI>k)|QIR?)`ijP{{AZ3l|+tqiJ(7&QFoX9R_}(qd6_M@R%4b~5JXoQe&jl)kSBJnRtkSC5Dmk;|7YFaQ7mSchw){_o^i00H(UpPT^zT6Hc)vBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
