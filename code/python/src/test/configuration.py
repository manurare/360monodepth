import os
import sys

# Add project library to Python path
python_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"Adding '{python_src_dir}' to sys.path")
sys.path.append(python_src_dir)  # /code/python/src/
sys.path.append(python_src_dir + "/utility/")  # /code/python/src/
sys.path.append(os.path.dirname(python_src_dir))  # /code/python/
sys.path.append(os.path.abspath(os.path.join(python_src_dir, os.pardir, os.pardir, "cpp/lib")))
# sys.path.append(os.path.join(python_src_dir, "../.."))  # CR: unused?

# Data directory /data/
# TODO: remove the trailing slash once all usages of TEST_DATA_DIR are updated
TEST_DATA_DIR = os.path.abspath("../../../data/") + "/"
MAIN_DATA_DIR = os.path.abspath("../../../data/") + "/"

# Set the PyTorch hub folder as an environment variable
# TODO: use os.path.join instead of string
os.environ['TORCH_HOME'] = TEST_DATA_DIR + 'models/'

# if __name__ =="__main__":
#     fnc = FileNameConvention()
#     fnc.set_filename_basename("++test++")
#     for attr in dir(fnc):
#         if not callable(getattr(fnc, attr)) and not attr.startswith("__"):
#             print(attr + ": \t" + getattr(fnc, attr))

#     print("===={}".format(fnc.subimage_depthmap_erp_aligned_filename_expression))
#     fnc.set_filepath_folder("d:/test/")
#     for attr in dir(fnc):
#         if not callable(getattr(fnc, attr)) and not attr.startswith("__"):
#             print(attr + ": \t" + getattr(fnc, attr))



import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',type=int, help='the test task index')

    args = parser.parse_args()
    return args