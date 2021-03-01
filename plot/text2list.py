text = """


[72.1 70.2 70.8 72.  70.3 69.2 70.7 71.9 71.1 71.8 71.2 71.1 69.8 71.6
 70.9 70.6 71.5 70.8 70.7 71.1]

"""

text = text.replace('\n', '')
text = text.replace('[', '')
text = text.replace(']', '')
vals = [float(val) for val in text.split()]
print(vals)
