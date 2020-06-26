import sys
import os
import pickle

class Pickle(object):
    def __init__(self):
        print("Pickle")
        self.obj_count = 0

    @staticmethod
    def save(obj, path, mode="wb"):
        """
        :param obj:  obj dict to dump
        :param path: save path
        :param mode:  file mode
        """
        print("save obj to {}".format(path))
        assert isinstance(obj, dict), "The type of obj must be a dict type."
        if os.path.exists(path):
            os.remove(path)
        pkl_file = open(path, mode=mode)
        pickle.dump(obj, pkl_file)
        pkl_file.close()

    @staticmethod
    def load(path, mode="rb"):
        """
        :param path:  pkl path
        :param mode: file mode
        :return: data dict
        """
        print("load obj from {}".format(path))
        if os.path.exists(path) is False:
            print("Path {} illegal.".format(path))
        pkl_file = open(path, mode=mode)
        data = pickle.load(pkl_file)
        pkl_file.close()
        return data

pcl = Pickle




