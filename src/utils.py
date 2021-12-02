# self.char_to_num = layers.experimental.preprocessing.StringLookup(
#     vocabulary=list(self.charList), num_oov_indices=0, mask_token=None, oov_token='[UNK]'
# )
# # Mapping integers back to original characters
# self.num_to_char = layers.experimental.preprocessing.StringLookup(
#     vocabulary=self.char_to_num.get_vocabulary(), num_oov_indices=0, oov_token='', mask_token=None, invert=True
# )
