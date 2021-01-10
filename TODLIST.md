## TODO LISt

1 - find what is causing the warning. I stopped finding yesterday after my deep lookout at the loss. The loss mechanism seems ok. However, it may comes from the laplacian loss used for d (unlikely though), or mabe the loss for 'ori'
    Alsp to take into account: custom L1 loss by lorenzo.

    -> Only happening before the loss for the training : Is there a reason for it not to happn for the val -> is the loss computation different for train and val ? : yes it is !!!
    ** It is either :
        * the laplacian loss
        * the ori loss

    => Not important by mathematicl standards, solving this just makes more lines of codes, is inefficient


2- Understand why I am getting worse results with the scene disposition.

