from text import cmudict

_punctuation = '!\'(),.:;? '
_special = '-'
_letters = 'abcdefghijklmnopqrstuvwxyz'

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = list(_special) + list(_punctuation) + list(_letters)  # + _arpabet
