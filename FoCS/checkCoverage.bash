#!/bin/bash
coverage erase
coverage run --branch -a DoubleLinkedList.py
coverage run --branch -a Lifo.py
coverage run --branch -a Queue.py
coverage run --branch -a TestLifoQueue.py
coverage report -m DoubleLinkedList.py Lifo.py Queue.py
