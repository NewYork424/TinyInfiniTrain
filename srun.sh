#!/bin/bash

srun --gres=gpu:nvidia:2 make test-cpp