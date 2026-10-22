# Import qkeras 
import qkeras

def get_quantizers(quantizer: 'QuantizationScheme'):
    # [@manuelbv]: OVERKILL, let's get ALL possible quantizers. I'm also adding some documentation here
    # cause this is a mess.
    # A list of the quantizers can be found here: https://github.com/google/qkeras/blob/master/notebook/QKerasTutorial.ipynb
    #     
    # Get params from quantizer
    m = quantizer.m # number of bits, argument "bits=..." in most quantizers
    n = quantizer.n # number of integer bits, argument "integer=..." in most quantizers
    keep_negative = quantizer.s # signed(1)/unsigned(0), argument "keep_negative=..." in most quantizers
    
    # Now, generally we set alpha to 1 or to auto, but look at this extract from the notebook in the link above:
    """
        When computing the scale in these quantizers, if alpha="auto", we compute the scale as a floating point 
        number. If alpha="auto_po2", we enforce the scale to be a power of 2, meaning that an actual 
        hardware or software implementation can be performed by just shifting the result of the 
        convolution or dense layer to the right or left by checking the sign of the scale (positive 
        shifts left, negative shifts right), and taking the log2 of the scale. This behavior is 
        compatible with shared exponent approaches, as it performs a shift adjustment to the channel.
    """
    alpha = 'auto_po2'

    # Now, let's get all possible quantizers
    qm = {}

    # Basic, logit quantizer
    qm['logit'] = qkeras.quantizers.quantized_bits(bits=m, integer=n, alpha=alpha,
                                                   keep_negative=keep_negative, 
                                                   symmetric=0, 
                                                   use_stochastic_rounding=False)
    # Just in case, replicate this also with the name "quantized_bits"
    qm['quantized_bits'] = qm['logit']
    
    # ReLU
    qm['relu'] = qkeras.quantizers.quantized_relu(bits=m, integer=n, use_sigmoid=0, 
                                                  use_stochastic_rounding=False)
    # Relu_po2
    qm['relu_po2'] = qkeras.quantizers.quantized_relu_po2(bits=m, 
                                                          max_value = 2**(2*m-1) if not keep_negative else 2**(2*m-2),
                                                         use_stochastic_rounding=False,
                                                         quadratic_approximation = False)
    qm['relu_po2_quadratic'] = qkeras.quantizers.quantized_relu_po2(bits=m, 
                                                          max_value = 2**(2*m-1) if not keep_negative else 2**(2*m-2),
                                                         use_stochastic_rounding=False,
                                                         quadratic_approximation = True)
    # uLaw
    qm['ulaw'] = qkeras.quantizers.quantized_ulaw(bits=m, integer=n, symmetric=0, u=255.0)
    # tanh
    qm['tanh'] = qkeras.quantizers.quantized_tanh(bits = m, use_stochastic_rounding = False, 
                                                  symmetric = False, use_real_tanh = False)
    # po2
    qm['po2'] = qkeras.quantizers.quantized_po2(bits=m, max_value=None, use_stochastic_rounding=False,
                                               quadratic_approximation=False)
    # smooth_sigmoid
    qm['smooth_sigmoid'] = qkeras.quantizers.smooth_sigmoid
    # hard_sigmoid
    qm['hard_sigmoid'] = qkeras.quantizers.hard_sigmoid
    # binary_sigmoid
    qm['binary_sigmoid'] = qkeras.quantizers.binary_sigmoid
    # smooth_tanh
    qm['smooth_tanh'] = qkeras.quantizers.smooth_tanh
    # hard_tanh
    qm['hard_tanh'] = qkeras.quantizers.hard_tanh
    # binary_tanh   
    qm['binary_tanh'] = qkeras.quantizers.binary_tanh

    # Other quantizers
    qm['bernoulli'] = qkeras.quantizers.bernoulli(alpha=alpha, temperature=6.0, use_real_sigmoid=True)
    qm['binary'] = qkeras.quantizers.binary(alpha=alpha, use_01=False, use_stochastic_rounding=False)
    qm['ternary'] = qkeras.quantizers.ternary(alpha=alpha, threshold=None, use_stochastic_rounding=False)
    qm['stochastic_binary'] = qkeras.quantizers.stochastic_binary(alpha=alpha, temperature=6.0, use_real_sigmoid=True)
    qm['stochastic_ternary'] = qkeras.quantizers.stochastic_ternary(alpha=alpha, threshold=None, temperature=8.0, use_real_sigmoid=True)
    
    return qm
