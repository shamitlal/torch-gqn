import numpy as np
import torch 

styledict = {1:"cream", 2:"gray", 3:"cream", 4:"light_purple", 5:"dark_green", 6:"cream", 7:"white", 8:"white", 9:"white", 10:"gray", 11:"light_brown", 12:"cream", 13:"black", 14:"dark_green", 15:"white", 16:"gray", 17:"gray", 19:"dark_green", 20:"gray", 21:"light_purple", 22:"dark_green", 23:"white", 24:"dark_green", 25:"white", 27:"gray", 31:"gray", 32:"cream", 35:"yellow", 36:"light_purple", 38:"dark_green", 39:"cream", 40:"dark_purple", 41:"gray", 42:"cream", 43:"white", 44:"dark_blue", 45:"black", 47:"unknown", 48:"cream", 49:"cream", 51:"cream", 52:"chocolate", 53:"silver", 54:"black", 55:"chocolate", 56:"dark_green", 61:"light_blue", 62:"unknown", 63:"light_green", 64:"gray", 65:"white", 66:"white", 67:"unknown", 68:"cream", 69:"gray", 71:"chocolate", 72:"gray", 73:"cream", 74:"cream", 75:"cream", 76:"dark_green", 77:"dark_green", 78:"dark_green", 79:"white", 80:"gray", 83:"cream", 84:"dark_green", 85:"white", 86:"white", 90:"darker_purple", 91:"white", 92:"gray", 93:"white", 94:"light_green", 95:"white", 96:"unknown", 98:"gray", 99:"unknown", 100:"gray", 103:"light_purple", 105:"white", 106:"white", 107:"white", 108:"light_chocolate", 109:"white", 110:"black", 111:"white", 113:"white", 115:"light_chocolate", 119:"unknown", 121:"unknown", 123:"light_green", 127:"white", 128:"white", 130:"unknwon", 131:"white", 136:"white", 140:"unknown", 141:"cream", 142:"gray", 143:"white", 144:"cream", 152:"light_green", 154:"light_chocolate", 157:"light_purple", 159:"white", 160:"unknown", 162:"dark_green", 166:"light_green", 167:"white", 170:"light_brown", 171:"unknown", 172:"gray", 175:"black", 176:"dark_green", 179:"black", 183:"cream", 184:"unknown", 185:"cream", 186:"light_purple", 187:"silver", 188:"light_purple", 194:"black", 195:"black", 196:"dark_brown", 198:"cream", 199:"unknown", 200:"silver", 202:"white", 204:"dark_brown", 208:"silver", 212:"white", 214:"cream", 220:"unknown", 222:"white", 223:"white", 230:"light_green", 231:"silver", 233:"white", 236:"light_green", 256:"dark_brown", 262:"silver", 263:"white", 269:"unknown", 351:"light_brown", 352:"white", 392:"unknown", 404:"dark_brown", 424:"light_brown", 451:"light_brown", 452:"light_brown", 454:"light_brown"}

def get_num_unique_styles(dictt):
    dictt = {v: k for k, v in dictt.items()}
    return len(dictt.keys())
