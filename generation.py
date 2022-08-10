import tensorflow as tf

one_step_reloaded = tf.saved_model.load('generation')


states = None
next_char = tf.constant(['#'])
result = [next_char]

for n in range(350):
  next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
