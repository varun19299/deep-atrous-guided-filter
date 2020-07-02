from torch.nn import init


def weights_init_identity(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        n_out, n_in, h, w = m.weight.data.size()

        # Last Layer
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return

        # Except Last Layer
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0

    elif classname.find("Conv1d") != -1:
        n_out, n_in, h = m.weight.data.size()

        # Except Last Layer
        m.weight.data.zero_()
        ch = h // 2
        for i in range(n_in):
            m.weight.data[i, i, ch] = 1.0

    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_identity_pixelshuffle(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        n_out, n_in, h, w = m.weight.data.size()

        for i in range(n_out):
            if i % 4 == 0:  # pixelshuffle ratio = 2
                init.xavier_uniform_(m.weight.data[i])
            else:
                m.weight.data[i] = m.weight.data[4 * (i // 4)]
        return

        # Last Layer
        # if n_out < n_in:
        #     for i in range(n_out):
        #         if i % 4 == 0:  # pixelshuffle ratio = 2
        #             init.xavier_uniform_(m.weight.data[i])
        #         else:
        #             m.weight.data[i] = m.weight.data[4 * (i // 4)]
        #     return

        # Except Last Layer
        # m.weight.data.zero_()
        # ch, cw = h // 2, w // 2
        # for i in range(n_in):
        #     m.weight.data[i, i, ch, cw] = 1.0


    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)
