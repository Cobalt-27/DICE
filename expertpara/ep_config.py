_config = {}

def get_config(key):
    return _config.get(key)

def set_config(key, value):
    _config[key] = value

# Usage example:
# set_config('key', 'value')
# print(get_config('key'))
# config.set('key', 'value')
# print(config.get('key'))