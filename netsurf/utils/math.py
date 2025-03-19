# Function to convert float to binary
# Author: Alan Guo
def float_to_binary(x, nbits):
    binary_rep = ''
    sign = 1 if x < 0 else 0
    x = 1 + x if sign == 1 else x
    for b in range(1, nbits):
        bit_value = 2.0**(-b)
        if x >= bit_value:
            binary_rep += '1'
            x -= bit_value
        else:
            binary_rep += '0'
    return str(sign) + binary_rep