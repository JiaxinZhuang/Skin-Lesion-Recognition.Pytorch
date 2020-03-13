"""Config.
All hyper paramerers and path can be changed in this file
"""

import sys
import argparse
# from utils.function import str2list, str2bool


class Config:
    """Config
    Attributes:
        parser: to read all config
        args: argument from argument parser
        config: save config in pairs like key:value
    """
    def __init__(self):
        """Load common and customized settings
        """
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(description='ISIC2018')
        self.config = {}

        # add setting via parser
        self._add_common_setting()
        self._add_customized_setting()
        # get argument parser
        self.args = self.parser.parse_args()
        # load them into config
        self._load_common_setting()
        self._load_customized_setting()
        # load config for specific server
        self._path_suitable_for_server()

    def _add_common_setting(self):
        # Need defined each time
        self.parser.add_argument('--experiment_index', default="None",
                                 type=str,
                                 help="001, 002, ...")
        self.parser.add_argument('--cudas', default='0', type=str,
                                 help="cuda visible device to use, maybe \
                                       parallel running, e.g., 1,3")
        self.parser.add_argument("--num_workers", default=4, type=int,
                                 help="num_workers of dataloader")
        # Log related
        self.parser.add_argument('--log_dir', default="./saved/logdirs/",
                                 type=str, help='store tensorboard files, \
                                 None means not to store')
        self.parser.add_argument('--model_dir', default="./saved/models/",
                                 type=str, help='store models, \
                                                 ../saved/models')
        # Hyper parameters
        self.parser.add_argument('--learning_rate', default=1e-4, type=float,
                                 help="lr")
        self.parser.add_argument("--batch_size", default=12, type=int,
                                 help="batch size of each epoch")
        self.parser.add_argument('--resume', default="", type=str,
                                 help="resume exp and epoch")
        self.parser.add_argument("--n_epochs", default=250, type=int,
                                 help="n epochs to train")

        self.parser.add_argument("--eval_frequency", default=1, type=int,
                                 help="Eval train and test frequency")

        self.parser.add_argument('--seed', default=47, type=int,
                                 help="Random seed for pytorch and Numpy ")
        self.parser.add_argument('--eps', default=1e-7, type=float,
                                 help="episilon for many formulation")
        self.parser.add_argument("--weight_decay", default=0, type=float,
                                 help="weight decay for optimizer")

        self.parser.add_argument("--optimizer", default="SGD", type=str,
                                 choices=["SGD", "Adam"],
                                 help="use SGD or Adam")
        # Input related
        # self.parser.add_argument("--input_size", default=300, type=int,
        #                          help="image input size for model")
        # self.parser.add_argument("--re_size", default=256, type=int,
        #                          help="resize to the size")
        self.parser.add_argument("--input_channel", default=3, type=int,
                                 help="channel of input image")
        # Backbone
        self.parser.add_argument("--backbone", default="resnet50", type=str,
                                 choices=["resnet50", "PNASNet5Large",
                                          "NASNetALarge"],
                                 help="backbone for model")
        # Initialization
        self.parser.add_argument("--initialization", default="pretrained",
                                 type=str,
                                 choices=["xavier_normal", "default",
                                          "pretrained", "kaiming_normal",
                                          "kaiming_uniform", "xavier_uniform"],
                                 help="initializatoin method")
        # self.parser.add_argument("--warmup_epochs", default=-1, type=int,
        #                          help="epochs to use warm up")

    def _add_customized_setting(self):
        """Add customized setting
        """
        self.parser.add_argument("--server", default="lab_center", type=str,
                                 choices=["local", "ls15", "ls16",
                                          "lab_center"],
                                 help="server to run the code")
        self.parser.add_argument("--num_classes", default=7, type=int,
                                 help="number of classes to classify.")
        self.parser.add_argument("--iter_fold", default=1, type=int,
                                 help="iter_fold to do cross validation.")
        self.parser.add_argument("--nproc_per_node", default=4, type=int,
                                 help="nproc per node.")
        self.parser.add_argument("--local_rank", default=0, type=int,
                                 help="gpu device")
        self.parser.add_argument("--loss_fn", default="WCE", type=str,
                                 choices=["CE", "WCE", "BCE"],
                                 help="Loss function for code.")

    def _load_common_setting(self):
        """Load default setting from Parser
        """
        self.config['experiment_index'] = self.args.experiment_index
        self.config['cudas'] = self.args.cudas
        self.config["num_workers"] = self.args.num_workers
        # Log
        self.config['log_dir'] = self.args.log_dir
        self.config['model_dir'] = self.args.model_dir
        # Hyper parameters
        self.config['learning_rate'] = self.args.learning_rate
        self.config['batch_size'] = self.args.batch_size
        self.config["resume"] = self.args.resume
        self.config['n_epochs'] = self.args.n_epochs
        self.config["eval_frequency"] = self.args.eval_frequency
        self.config['seed'] = self.args.seed
        self.config["eps"] = self.args.eps
        self.config["weight_decay"] = self.args.weight_decay
        self.config["optimizer"] = self.args.optimizer
        # Input
        # self.config["input_size"] = self.args.input_size
        # self.config["re_size"] = self.args.re_size
        # Backbone
        self.config["backbone"] = self.args.backbone
        # Intialization
        self.config["initialization"] = self.args.initialization
        # self.config["warmup_epochs"] = self.args.warmup_epochs

    def _load_customized_setting(self):
        """Load sepcial setting
        """
        self.config["server"] = self.args.server
        self.config["num_classes"] = self.args.num_classes
        self.config["iter_fold"] = self.args.iter_fold
        self.config["nproc_per_node"] = self.args.nproc_per_node
        self.config["local_rank"] = self.args.local_rank
        self.config["loss_fn"] = self.args.loss_fn

    def _path_suitable_for_server(self):
        """Path suitable for server
        """
        if self.config["server"] == "desktop":
            self.config["log_dir"] = "/home/lincolnzjx/Desktop/ISIC_2018_Classification/saved/logdirs"
            self.config["model_dir"] = "/home/lincolnzjx/Desktop/ISIC_2018_Classification/saved/models"
        if self.config["server"] == "local":
            self.config["log_dir"] = "/media/lincolnzjx/Disk21/ISIC_2018_Classification/saved/logdirs"
            self.config["model_dir"] = "/media/lincolnzjx/Disk21/ISIC_2018_Classification/saved/models"
        elif self.config["server"] == "ls15":
            self.config["log_dir"] = "/data15/jiaxin/ISIC_2018_Classification/saved/logdirs"
            self.config["model_dir"] = "/data15/jiaxin/ISIC_2018_Classification/saved/models"
        elif self.config["server"] == "ls16":
            self.config["log_dir"] = "/data16/jiaxin/ISIC_2018_Classification/saved/logdirs"
            self.config["model_dir"] = "/data16/jiaxin/ISIC_2018_Classification/saved/models"
        elif self.config["server"] == "lab_center":
            self.config["log_dir"] = "./saved/logdirs"
            self.config["model_dir"] = "./saved/models"
        else:
            print("Illegal server configuration")
            sys.exit(-1)

    def print_config(self, _print=None):
        """print config
        """
        _print("==================== basic setting start ====================")
        for arg in self.config:
            _print('{:20}: {}'.format(arg, self.config[arg]))
        _print("==================== basic setting end ====================")

    def get_config(self):
        """return config
        """
        return self.config
