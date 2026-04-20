#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Laura
# =============================================================================
"""This file contains some basic operations to make other scripts more efficient"""
# =============================================================================
# Imports
# =============================================================================
import json


def read_json(file_name):

    with open(file_name, 'r') as jsn:
        file_contents = json.load(jsn)

    return file_contents


def write_json(file_name, json_dict):

    with open(file_name, 'w') as jsn:
        json.dump(json_dict, jsn)