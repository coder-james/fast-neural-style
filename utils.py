import tensorflow as tf
import yaml
from os import listdir
from os.path import isfile, join

slim = tf.contrib.slim

def get_image(filepath, height, width, preprocess_fn, queue=None):
    png = filepath.lower().endswith('png')
    if queue is None:
      img_bytes = tf.read_file(filepath)
    else:
      reader = tf.WholeFileReader()
      _, img_bytes = reader.read(queue)

    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)
    return preprocess_fn(image, height, width)

def batch_image(batch_size, height, width, path, preprocess_fn, epochs=2, shuffle=True):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    if not shuffle:
        filenames = sorted(filenames)

    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)

    processed_image = get_image(filenames[0], height, width, preprocess_fn, queue=filename_queue)
    return tf.train.batch([processed_image], batch_size, dynamic_pad=True)

def _get_init_fn(FLAGS):
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS


if __name__ == '__main__':
    f = read_conf_file('conf/chritmas.yml')
    print(f.loss_model_file)
