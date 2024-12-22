import os
import torch
import logging
import numpy as np
import methods
import time
from models.model import get_model
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual"                    # sequence of gradually increasing / decreasing domain shifts
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    errs_stu = []
    errs_tea = []
    errs_5_stu = []
    errs_5_tea = []
    domain_dict = {}

    # start evaluation
    start_time = time.time()
    for i_dom, domain_name in enumerate(domain_seq_loop):
        start_time_cur_domain = time.time()
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               preprocess=model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               domain_names_all=domain_sequence,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))


            # evaluate the model
            accs, domain_dict, num_samples = get_accuracy(model,
                                                         data_loader=test_data_loader,
                                                         dataset_name=cfg.CORRUPTION.DATASET,
                                                         domain_name=domain_name,
                                                         setting=cfg.SETTING,
                                                         domain_dict=domain_dict,
                                                         print_every=cfg.PRINT_EVERY,
                                                         device=device)

            if len(accs) > 1:
                acc = accs[0]
                accuracy_stu = accs[1]
                accuracy_tea = accs[2]
            else:
                acc = accs[0]
                accuracy_stu = None
                accuracy_tea = None
            err = 1. - acc
            errs.append(err)
            if accuracy_stu is not None:
                errs_stu.append(1. - accuracy_stu)
                errs_tea.append(1. - accuracy_tea)

            if severity == 5 and domain_name != "none":
                errs_5.append(err)
                if accuracy_stu is not None:
                    errs_5_stu.append(1. - accuracy_stu)
                    errs_5_tea.append(1. - accuracy_tea)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")
        end_time_cur_domain = time.time()
        total_time_cur_domain  = end_time_cur_domain - start_time_cur_domain
        # print(f"cur_domain running time: {total_time_cur_domain} seconds")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"total running time: {total_time} seconds")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
        if len(errs_stu) > 0:
            logger.info(f"stu mean error: {np.mean(errs_stu):.2%}, mean error at 5: {np.mean(errs_5_stu):.2%}")
            logger.info(
                f"tea mean error: {np.mean(errs_tea):.2%}, mean error at 5: {np.mean(errs_5_tea):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    formatted_errs = [f"{err:.2%}" for err in errs]
    error_string = " ".join(formatted_errs)
    logger.info(f"all error: {error_string}")

    formatted_errs_stu = [f"{err:.2%}" for err in errs_stu]
    error_string = " ".join(formatted_errs_stu)
    logger.info(f"stu all error: {error_string}")

    formatted_errs_tea = [f"{err:.2%}" for err in errs_tea]
    error_string = " ".join(formatted_errs_tea)
    logger.info(f"tea all error: {error_string}")


if __name__ == '__main__':
    evaluate('"Evaluation.')
