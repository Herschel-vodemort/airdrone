"""
main function implementation
"""
from result_generate import generate_result_multi_p, result_merge
from evaluator import result_evaluator


def main():
    # generate trajectory
    generate_result_multi_p()
    # evaluate result
    result_evaluator()
    result_merge()
    return


if __name__ == '__main__':
    main()