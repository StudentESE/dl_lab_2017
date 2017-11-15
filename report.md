# Report

Due to time presure (later starting) I didn't finish in time.

> Reports:
>  Short 1-2 page report explaining your results, typically accompanied by 1-2 figures (e.g. a learning curve)
>  hand in the code you used to generate these results as well, send us your github/bitbucket repo

## 3 Layers

There are three fully connected layers with differing gradients:

```
checking gradient for layer 1
diff 1.30e-07
diff 4.62e-08
checking gradient for layer 2
diff 9.07e-08
diff 2.36e-08
checking gradient for layer 3
diff 6.10e-08
diff 3.59e-08
```

## Training vs. Validation Error

After about 7 epochs also the validation error was less than 3% which was the job to do.

> Epoch: 7 Validation: loss 0.0985, **Validation error 0.0299**
> epoch 8, loss 0.0584, **train error 0.0176**

![Traing vs Validation Errors](https://github.com/StudentESE/dl_lab_2017/blob/master/trainingcurve.png)

## Errors per Figrue

Different figures have depending on their similarity of some characteristics (like parts of a curve or line) more or less errors in detection.

> Test error: 0.02620
> Errors per figure:  [ 10.   8.  29.  20.  52.  24.  24.  32.  37.  26.]
> Number 1 has best matchings and 4 was the most incorrectly classified figure

For example this both image are representing the numbers figure 4 but also could be a 9 or a 7.

![4](https://github.com/StudentESE/dl_lab_2017/blob/master/4.png)
![4](https://github.com/StudentESE/dl_lab_2017/blob/master/4b.png)
