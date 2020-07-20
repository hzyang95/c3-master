import json

from data.data_instance import ReadState
from general_util.logger import get_child_logger
from general_util.register import reader_dict as registered_readers
from reader.race_multiple_reader import RACEMultipleReader
from reader.race_reader import RACEReader



logger = get_child_logger(__name__)

try:
    from reader.race_multiple_reader_roberta import RACEMultipleReader as RACEMultipleReaderRoberta
except ImportError:
    logger.warn("Couldn't load models with roBERTa")
    RACEMultipleReaderRoberta = None

reader_dict = {
    'race': RACEReader,
    'multiple-race': RACEMultipleReader,
    'multiple-race-roberta': RACEMultipleReaderRoberta,
}
reader_dict.update(registered_readers)


def initialize_reader(name: str, *arg, **kwargs):
    logger.info('Loading reader {} ...'.format(name))
    return reader_dict[name](*arg, **kwargs)


def prepare_read_params(args):
    if args.reader_name in ['race', 'multiple-race']:
        read_params = {}
    else:
        raise RuntimeError(f'Wrong reader_name for {args.reader_name}')

    return read_params


def read_from_squad(input_file: str):
    with open(input_file, 'r') as f:
        data = json.load(f)['data']

    contexts = data['contexts']
    output = []
    for context in contexts:
        output.extend(context)
    return output
