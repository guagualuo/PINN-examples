import torch
import torch.nn as nn


class Dense(nn.Module):
    def __init__(self, activation, inputs, outputs):
        super(Dense, self).__init__()
        self.linear = nn.Linear(inputs, outputs)
        self.activation = activation
    
    def forward(self, x):
        return self.activation(self.linear(x))


class FFN(nn.Module):
    def __init__(self, activation, n_hidden, n_nodes, inputs):
        super(FFN, self).__init__()
        layers = []
        for i in range(n_hidden+2):
            if i == 0:
                layers.append(Dense(activation, inputs, n_nodes))
            elif i == n_hidden+1:
                layers.append(nn.Linear(n_nodes, 1))
            else:
                layers.append(Dense(activation, n_nodes, n_nodes))
        self.layers = nn.ModuleList(layers)


class StationaryFFN(FFN):
    def __init__(self, activation, n_hidden, n_nodes, inputs):
        super(StationaryFFN, self).__init__(activation, n_hidden, n_nodes, inputs)
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

class TimeDependentFFN(FFN):
    def __init__(self, activation, n_hidden, n_nodes, inputs):
        super(TimeDependentFFN, self).__init__(activation, n_hidden, n_nodes, inputs+1)
        
    def forward(self, x, t):
        xt = torch.cat([x, t], axis=1)
        for l in self.layers:
            xt = l(xt)
        return xt


class Callback():
    def __init__(self): pass
    def on_train_begin(self, **kwargs): pass
    def on_train_end(self, **kwargs): pass
    def on_epoch_begin(self, **kwargs): pass
    def on_epoch_end(self, **kwargs): pass
    def on_batch_begin(self, **kwargs): pass
    def on_batch_end(self, **kwargs): pass
    def on_loss_begin(self, **kwargs): pass
    def on_loss_end(self, **kwargs): pass
    def on_step_begin(self, **kwargs): pass
    def on_step_end(self, **kwargs): pass

class ResamplingHandler(Callback):
    def __init__(self, trainer, sampling_rate):
        super(ResamplingHandler, self).__init__()
        self.sampling_rate = sampling_rate
        self.trainer = trainer
        
    def on_batch_begin(self, **kwargs):
        if self.trainer.iteration % self.sampling_rate == 0:
            self.trainer.generate_samples()


class ValueTracker(Callback):
    def __init__(self, name, values=[]):
        super(ValueTracker, self).__init__()
        self.values = values
        self.name = name
        
    def on_loss_end(self, **kwargs):
        if self.name in kwargs.keys():
            self.values.append(kwargs[self.name])
        

class ValidationErrorTracker(Callback):
    def __init__(self, trainer, metric, validation, errors=[]):
        super(ValidationErrorTracker, self).__init__()
        self.errors = errors
        self.trainer = trainer
        self.metric = metric
        self.valid_x, self.valid_t, self.valid_u = validation
    def on_loss_end(self, **kwargs):
        self.errors.append(self.metric(self.trainer.net(self.valid_x, self.valid_t), self.valid_u))
    

class DiagnosticVerbose(Callback):
    def __init__(self, trainer, print_rate, values):
        super(DiagnosticVerbose, self).__init__()
        self.trainer = trainer
        self.print_rate = print_rate
        self.values = values
        
    def on_step_end(self, **kwargs):
        if self.trainer.iteration % self.print_rate == 0:
            message = f"At epoch {self.trainer.iteration}: "
            for key, val in self.values.items():
                message += f"{key}: {val[-1]:.4f}; "
            print(message)
