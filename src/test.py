import logging

from src.utils import evaluate_dataset_baseline


def test(
    test_dataset,
    model_A,
    model_B,
    device,
    distance_fn,
    args,
    dictionary=None,
    save_performances=False,
    best_beta=None,
):
    logger = logging.getLogger()
    model_A.eval()
    model_B.eval()

    test_evaluation = _get_test_performance(
        test_dataset=test_dataset,
        model_B=model_B,
        device=device,
        distance_fn=distance_fn,
        args=args,
        dictionary=dictionary,
        save_performances=save_performances,
        best_beta=best_beta,
    )

    if (
        args.dataset_name == "VGGSound"
        or args.dataset_name == "UCF"
        or args.dataset_name == "ActivityNet"
    ):
        output_string = rf"""
                    Seen performance={100 * test_evaluation["both"]["seen"]:.2f}, Novel performance={100 * test_evaluation["both"]["unseen"]:.2f}, GFSL performance={100 * test_evaluation["both"]["hm"]:.2f}, FSL performance={100 * test_evaluation["both"]["zsl"]:.2f} 
                    """
    else:
        raise NotImplementedError()

    logger.info(output_string)

    logger.info(
        rf"""Discriminator seen={100*test_evaluation['both']['correct_seen']:.2f}, Discriminator unseen={100*test_evaluation['both']['correct_unseen']:.2f}"""
    )

    return test_evaluation


def _get_test_performance(
    test_dataset,
    model_B,
    device,
    distance_fn,
    args,
    dictionary=None,
    save_performances=False,
    best_beta=None,
):
    logger = logging.getLogger()

    best_beta_combined = best_beta
    logger.info(f"Best beta combined: {best_beta_combined}")

    test_evaluation = evaluate_dataset_baseline(
        test_dataset,
        model_B,
        device,
        best_beta=best_beta_combined,
        args=args,
        dictionary=dictionary,
        save_performances=save_performances,
    )

    return test_evaluation
