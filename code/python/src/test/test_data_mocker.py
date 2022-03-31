import configuration as config

from utility import data_mocker

def test_create_alignment_data():
    pass


def test_data_visualizing():
    data_root_dir = "D:/workspace_windows/InstaOmniDepth/InstaOmniDepth_github/code/cpp/bin/Release/"
    frame_number = 3
    data_mocker.data_visualizing(data_root_dir, frame_number)


if __name__ == "__main__":

    args = config.get_parser()

    test_list = []
    test_list.append(args.task)

    if 1 in test_list:
        test_create_alignment_data()
    if 2 in test_list:
        test_data_visualizing()
