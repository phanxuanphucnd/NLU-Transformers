# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

def print_denver(message, denver_version):
    print("")
    print('\n'.join([
        '▅ ▆ ▇ █ Ⓓ ⓔ ⓝ ⓥ ⓔ ⓡ  █ ▇ ▆ ▅ {}'.format(denver_version), 
        ''
    ]))

def print_style_free(message, print_fun=print):
    print_fun("")
    print_fun("░▒▓█  {}".format(message))

def print_style_time(message, print_fun=print):
    print_fun("")
    print_fun("⏰  {}".format(message))
    print_fun("")
    
def print_style_warning(message, print_fun=print):
    print_fun("")
    print_fun("⛔️  {}".format(message))
    print_fun("")
    
def print_style_notice(message, print_fun=print):
    print_fun("")
    print_fun("📌  {}".format(message))
    print_fun("")

def print_line(text, print_fun=print):
    print_fun("")
    print_fun("➖➖➖➖➖➖➖➖➖➖ {} ➖➖➖➖➖➖➖➖➖➖".format(text.upper()))
    print_fun("")
