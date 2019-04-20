import tensorflow as tf
import pickle
from Transformer import Transformer
from Utils import iter_data, Logger
import time


def save(model, train_loss_results, validation_loss_results, cnt):
    with open('train_loss_results.pkl', 'wb') as pkl:
        pickle.dump(train_loss_results, pkl)

    with open('validation_loss_results.pkl', 'wb') as pkl:
        pickle.dump(validation_loss_results, pkl)

    model.save_weights("./checkpoints/cp-{}.ckpt".format(format(cnt)))


def format(x, max_len=4):
    x = str(x)
    return "0" * (max_len - len(x)) + x


def STLR(t, cut_frac=0.075, ratio=24, lr_max=0.002, T=17500):
    cut = int(T * cut_frac)
    p = t / cut if t < cut else 1 - (t - cut) / (cut * (ratio - 1))
    lr = lr_max * (1 + p * (ratio - 1)) / ratio
    return lr


def decay(lr, t):
    lr -= lr * (1 / (t ** 0.5))
    return lr


def train(model, learning_rate=0.00025, n_epochs=100, n_batch=64, n_ctx=512,
          train_steps=100, validation_steps=2000, save_steps=2000, log_path='train.log',
          load_path = "./checkpoints/cp-0000.ckpt", lr_fn = 'STLR'):
    X = tf.placeholder(tf.int32, [None, n_ctx, 2])
    M = tf.placeholder(tf.int32, [None, n_ctx])
    logits, losses = model([X, M])

    lr = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    opt = optimizer.minimize(losses)

    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
    model.load_weights(load_path)

    train_losses = []
    train_loss_results = []
    validation_loss_results = []

    step = 0
    cnt = 4
    train_generator = iter_data(n_batch, n_epochs)
    logger = Logger(path=log_path)
    start = time.time()

    for tokens, masks in train_generator:
        step += 1
        if lr_fn == 'STLR':
            _, train_logits, train_loss = sess.run([opt, logits, losses], {X: tokens, M: masks, lr: STLR(step)})

        else:
            _, train_logits, train_loss = sess.run([opt, logits, losses], {X: tokens, M: masks, lr: learning_rate})

        train_losses.append(train_loss)

        if step % train_steps == 0:
            if lr_fn != 'STLR':
                learning_rate = decay(learning_rate, step)

            train_loss_results.append(sum(train_losses) / len(train_losses))
            train_losses = []
            logger.log(step=step, train_loss=train_loss_results[-1], time=time.time() - start)

        if step % validation_steps == 0:
            validation_generator = iter_data(n_batch, train=False)
            validation_losses = []
            for validation_tokens, validation_masks in validation_generator:
                validation_losses.append(sess.run(losses, {X: validation_tokens, M: validation_masks}))

            validation_loss_results.append(sum(validation_losses) / len(validation_losses))
            logger.log(step=step, validation_loss=validation_loss_results[-1], time=time.time() - start)

        if step % save_steps == 0:
            cnt += 1
            save(model, train_loss_results, validation_loss_results, cnt)


if __name__ == '__main__':
    model = Transformer("Model", 40478)
    train(model, n_batch=8, learning_rate=0.0001)
