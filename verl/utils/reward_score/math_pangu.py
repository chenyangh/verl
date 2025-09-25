# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import sys
import logging
import re
import json


from .pangu_math.parser import parse as mparse
from .pangu_math.grader import verify as mcheck


def format_boxed_check(model_resp_answer: str) -> bool:
    # '//boxed' existency check
    if "\\boxed" not in model_resp_answer:
        return False
    else:
        ans = model_resp_answer.split("\\boxed")[-1]
        if not ans or ans[0] != "{":
            return False
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
        if a == "" or stack > 0:
            return False

    return True


def parse_model_response(resp: str):
    # unify the chinese colon
    resp = resp.replace("：", ": ")
    resp = resp.replace("，", ", ")
    resp = resp.replace("、", ", ")
    resp = resp.replace("。", ". ")
    resp = resp.replace("？", "? ")
    resp = resp.replace("！", "! ")
    resp = resp.replace("”", "\"")
    resp = resp.replace("“", "\"")
    resp = resp.replace("（", "(")
    resp = resp.replace("）", ")")
    resp = resp.replace("【", "[")
    resp = resp.replace("】", "]")

    return resp


def correctness_check_mv(model_resp: str, gt: str, answer_tag: int):
    pred = mparse(parse_model_response(model_resp), skip_unit=False if answer_tag == 1 else True)
    parsed_gt = mparse(str(gt), skip_unit=False if answer_tag == 1 else True)
    return mcheck(parsed_gt, pred)



def compute_math_reward(solution_str, ground_truth, answer_tag=1) -> float:
    model_resp = solution_str
    model_resp_answer = parse_model_response(model_resp)
    flag_box_format = format_boxed_check(model_resp_answer)
 
    # reward shaping
    if not flag_box_format:
        score = -1.0
    else:
        flag_correctness = correctness_check_mv(model_resp_answer, ground_truth, answer_tag)

        correctness_score = 1.0 if flag_correctness else -0.5
        score = 0.0 + correctness_score
    return score