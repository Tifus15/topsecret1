import yaml
#from device_util import ROOT_PATH





def open_yamls(name):
    with open(ROOT_PATH+"/src/yamls/"+name) as stream:
        try:
            data_dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data_dict