from collections import deque
import numpy as np

class ArrayFormatter(object):
    def __init__(self, array, indent=0, array_indent=2, name='unnamed array'):
        self.array        = array
        self.indent       = indent
        self.array_indent = array_indent
        self.name         = name

    def __str__(self):
        indent_str = ' '*self.indent
        array_indent_str = ' '*self.array_indent

        rows = []
        rows.append('{}{} {}:'.format(indent_str, self.name, self.array.shape))

        if self.array.ndim == 1:
            self.array       = np.copy(self.array)
            self.array.shape = (1, self.array.shape[0])

        nr,nc = self.array.shape
        for ridx in range(nr):
            vals = deque([self._format(elem) for elem in self.array[ridx,:]])
            if (ridx == 0) and (ridx == nr-1):
                vals.extendleft(['[ ['])
                vals.extend(['] ]'])
            elif ridx == 0:
                vals.extendleft(['[ ['])
                vals.extend([']  '])
            elif ridx == nr-1:
                vals.extendleft(['  ['])
                vals.extend(['] ]'])
            else:
                vals.extendleft(['  ['])
                vals.extend([']  '])
            row = ' '.join(vals)
            row = '{}{}{}'.format(indent_str, array_indent_str, row)
            rows.append(row)
        return '\n'.join(rows)
    def __repr__(self):
        return self.__str__()

class BooleanArrayFormatter(ArrayFormatter):

    def __init__(self, tf_only=True, **kwargs):
        super(BooleanArrayFormatter, self).__init__(**kwargs)
        self.tf_only = tf_only

    def _format(self, elem):
        if self.tf_only:
            result = 'T' if elem else 'F'
        else:
            result = 'True' if elem else 'False'
        return result


class FloatArrayFormatter(ArrayFormatter):

    def __init__(self, format='{:>+10.3e}', **kwargs):
        super(FloatArrayFormatter, self).__init__(**kwargs)
        self.format = format

    def _format(self, elem):
        return self.format.format(elem)
