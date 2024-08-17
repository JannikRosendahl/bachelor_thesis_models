# meant to be imported by notebooks
# contains utility functions for data processing and visualization
# extended as experiments advance, meant to be backwards compatible
import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk_model_selection
import sklearn.utils as sk_utils

# define sklearn.utils.shuffle with a fixed random_state
def shuffle(*args, **kwargs):
    # remove random_state from kwargs
    kwargs.pop('random_state', None)
    return sk_utils.shuffle(*args, random_state=42, **kwargs)

# define sklearn.sklearn.model_selection.train_test_split with a fixed random_state
def train_test_split(*args, **kwargs):
    # remove random_state from kwargs
    kwargs.pop('random_state', None)
    return sk_model_selection.train_test_split(*args, random_state=42, **kwargs)


"""
general utilities
"""

def class_from_filename(filename):
    """
    Files have format: {label}_{subject_uuid}_{occurrence_id}.txt
    E.g. "adjkerntz_77C420A3-3B91-11E8-B8CE-15D78AC88FB6_2.txt".
    Note that the label may contain underscores (pwd_mkdb).
    """
    if filename.count('_') == 2:
        label, _, _ = filename.split('_')
    elif filename.count('_') == 3:
        label, tmp, _, _ = filename.split('_')
        label += '_' + tmp
    else:
        print(f'Unknown file name format: {filename}')
        raise ValueError()
    return label


"""
preprocessing utilities
notes:
    - the string 'None' is used to represent the absence of a value
    - the string 'Other' is used to represent a catch-all category
"""

def preprocess_path_to_tld(path):
    if path != 'None' and not path.startswith('/'):
        path = 'Other'
    if path != 'None' and path != 'Other':
        path = path.split('/')[1]
    return path

private_ip_ranges = [
    [ '10.0.0.0', '10.255.255.255' ],
    [ '100.64.0.0', '100.127.255.255' ],
    [ '172.16.0.0', '172.31.255.255' ],
    [ '192.0.0.0', '192.0.0.255' ],
    [ '192.168.0.0', '192.168.255.255' ],
    [ '198.18.0.0', '198.19.255.255' ],
]
loopback_ip_range = [ '127.0.0.0', '127.255.255.255' ]
local_network_ip_range = [ '0.0.0.0', '0.255.255.255' ]
internet_ip_ranges = [
    [ '224.0.0.0', '239.255.255.255' ],
    [ '240.0.0.0', '255.255.255.254' ],
]

def ip_to_int(ip):
    return sum([int(octet) * 256 ** i for i, octet in enumerate(reversed(ip.split('.')))])

def ip_in_range(ip, ip_range):
    """
    Checks if an IP is in a given range.
    A range is a list of two strings, each representing lower and upper bound of the range.
    """
    ip = ip_to_int(ip)
    lower_bound = ip_to_int(ip_range[0])
    upper_bound = ip_to_int(ip_range[1])
    return lower_bound <= ip <= upper_bound

def preprocess_ip(ip):
    if ip == 'None':
        return ip
    if ip == 'localhost':
        return 'Loopback'
    try:
        ip_to_int(ip)
    except ValueError:
        return 'Other'
    for range in private_ip_ranges:
        if ip_in_range(ip, range):
            return 'Private'
    if ip_in_range(ip, loopback_ip_range):
        return 'Loopback'
    if ip_in_range(ip, local_network_ip_range):
        return 'Local'
    for range in internet_ip_ranges:
        if ip_in_range(ip, range):
            return 'Internet'
    return 'Other'

ports_to_categories = {
    1: 'TCPMUX',
    22: 'SSH',
    25: 'SMTP',
    53: 'DNS',
    80: 'HTTP',
    143: 'IMAP',
    512: 'EXEC',
}

def preprocess_port(port):
    if port == 'None':
        return port
    port = int(port)
    return ports_to_categories.get(port, 'Other')


"""
data loading utilities
"""

class Preprocessor:
    """
    Class used to preprocess lines into vectors of preprocessed and vectorized values.
    """
    def __init__(self, enabled_features: list) -> None:
        self.enabled_features = enabled_features

        self.event_types_map = {}
        self.users_map = {}
        self.filetypes_map = {'None': 0}
        self.path_map = {'None': 0}
        self.addr_map = {'None': 0}
        self.port_map = {'None': 0}

    def process(self, line: list) -> dict:
        """
        Processes a line from the dataset.
        Returns a dictionary with the processed values.
        Values are vectorized and mapped to integers.
        """
        line = line.strip('\n ').split(',')

        v = {}

        if 'TYPE' in self.enabled_features:
            assert len(line) >= 1
            v['TYPE'] = self.event_types_map.setdefault(line[0], len(self.event_types_map))

        if 'USERNAME' in self.enabled_features:
            assert len(line) >= 2
            v['USERNAME'] = self.users_map.setdefault(line[1], len(self.users_map))

        if 'PRED_OBJ_TYPES' in self.enabled_features:
            assert len(line) >= 4
            v['PRED_OBJ1_TYPE'] = self.filetypes_map.setdefault(line[2], len(self.filetypes_map))
            v['PRED_OBJ2_TYPE'] = self.filetypes_map.setdefault(line[3], len(self.filetypes_map))

        if 'PRED_OBJ_PATHS' in self.enabled_features:
            assert len(line) >= 6
            v['PRED_OBJ1_PATH'] = self.path_map.setdefault(preprocess_path_to_tld(line[4]), len(self.path_map))
            v['PRED_OBJ2_PATH'] = self.path_map.setdefault(preprocess_path_to_tld(line[5]), len(self.path_map))

        if 'PRED_OBJ_NETINFO' in self.enabled_features:
            assert len(line) >= 14
            v['PRED_OBJ1_LOCALIP'] = self.addr_map.setdefault(preprocess_ip(line[6]), len(self.addr_map))
            v['PRED_OBJ1_LOCALPORT'] = self.port_map.setdefault(preprocess_port(line[7]), len(self.port_map))
            v['PRED_OBJ1_REMOTEIP'] = self.addr_map.setdefault(preprocess_ip(line[8]), len(self.addr_map))
            v['PRED_OBJ1_REMOTEPORT'] = self.port_map.setdefault(preprocess_port(line[9]), len(self.port_map))
            v['PRED_OBJ2_LOCALIP'] = self.addr_map.setdefault(preprocess_ip(line[10]), len(self.addr_map))
            v['PRED_OBJ2_LOCALPORT'] = self.port_map.setdefault(preprocess_port(line[11]), len(self.port_map))
            v['PRED_OBJ2_REMOTEIP'] = self.addr_map.setdefault(preprocess_ip(line[12]), len(self.addr_map))
            v['PRED_OBJ2_REMOTEPORT'] = self.port_map.setdefault(preprocess_port(line[13]), len(self.port_map))

        return v


"""
ML utilities
"""
class Generator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, fixed_length, **kwargs):
        # valid **kwargs: workers, use_multiprocessing, max_queue_size
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.fixed_length = fixed_length
        self.no_samples = len(X)
        self.no_batches = int(np.ceil(self.no_samples / self.batch_size))

    def __len__(self):
        return self.no_batches

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, self.no_samples)

        # Get the batch data
        X_batch = self.X[start:end]
        y_batch = self.y[start:end]

        # Pad or truncate each sequence in X_batch to the fixed length
        X_batch_fixed = [self._pad_or_truncate(x, self.fixed_length) for x in X_batch]

        # Convert to numpy arrays
        X_batch_fixed = np.array(X_batch_fixed)
        y_batch = np.array(y_batch)

        return X_batch_fixed, y_batch

    def _pad_or_truncate(self, sequence, length):
        return np.pad(sequence, ((0, length - len(sequence)), (0, 0)), mode='constant', constant_values=0)

    def on_epoch_end(self):
        # shuffle data
        self.X, self.y = shuffle(self.X, self.y, random_state=42)