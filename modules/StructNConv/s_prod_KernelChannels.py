
import torch
import torch.nn.functional as F

from modules.StructNConv.KernelChannels import KernelChannels
from modules.StructNConv.retrieve_indices import retrieve_indices


class s_prod_KernelChannels(torch.nn.Module):
    def __init__(self,  kernel_size, stride=1, padding=0, dilation=1):
        super(s_prod_KernelChannels, self).__init__()
        self.kernel_size = kernel_size
        self.dilated_kernel_size = 1+(kernel_size-1)*dilation
        self.stride = stride
        self.dilation = dilation
        self.kernel_channels = KernelChannels(1+(kernel_size-1)*dilation, 1, padding, 1)

    def forward(self, s, cs):
        '''
        :param
        s, cs to unroll, 4 dimensional
        :return:
        5 dimensional, neighbouring elements as additional channels in dim 2 (of 0 to 4)
        [[[[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]]] -> [1, 2, 3, 4, 5, 6, 7, 8, 9] for middle element with kernel size 3
        s of middle = 1, cs of middle = 1
        cs_out = product of cs along path to middle (including middle) with highest cs
        s_out = product of s along same path
        '''
        s_roll = self.kernel_channels.kernel_channels(s)
        cs_roll = self.kernel_channels.kernel_channels(cs)
        s_out = torch.ones_like(s_roll)
        cs_out = torch.ones_like(s_roll)

        mid = int((self.dilated_kernel_size-1)/2)*self.dilated_kernel_size+int((self.dilated_kernel_size-1)/2)
        mid_left_up = mid
        mid_right_up = mid + (self.dilated_kernel_size+1) % 2
        mid_left_down = mid + self.dilated_kernel_size*(self.dilated_kernel_size+1) % 2
        mid_right_down = mid + (1+self.dilated_kernel_size) * (self.dilated_kernel_size+1) % 2

        if (self.dilated_kernel_size+1) % 2 == 0:
            s_out[:, :, mid_left_up, :, :] = s_roll[:, :, mid_left_up, :, :]
            cs_out[:, :, mid_left_up, :, :] = cs_roll[:, :, mid_left_up, :, :]
            s_out[:, :, mid_right_up, :, :] = s_roll[:, :, mid_right_up, :, :]
            cs_out[:, :, mid_right_up, :, :] = cs_roll[:, :, mid_right_up, :, :]
            s_out[:, :, mid_left_down, :, :] = s_roll[:, :, mid_left_down, :, :]
            cs_out[:, :, mid_left_down, :, :] = cs_roll[:, :, mid_left_down, :, :]
            s_out[:, :, mid_right_down, :, :] = s_roll[:, :, mid_right_down, :, :]
            cs_out[:, :, mid_right_down, :, :] = cs_roll[:, :, mid_right_down, :, :]

            if self.dilated_kernel_size > 2:
                c = mid_left_up - self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_up, :, :]*cs_roll[:, :, c, :, :]
                c = mid_left_up - self.dilated_kernel_size - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_up, :, :]*cs_roll[:, :, c, :, :]
                c = mid_left_up - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_up, :, :]*cs_roll[:, :, c, :, :]

                c = mid_right_up - self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_up, :, :]*cs_roll[:, :, c, :, :]
                c = mid_right_up - self.dilated_kernel_size + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_up, :, :]*cs_roll[:, :, c, :, :]
                c = mid_right_up + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_up, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_up, :, :]*cs_roll[:, :, c, :, :]

                c = mid_left_down + self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_down, :, :]*cs_roll[:, :, c, :, :]
                c = mid_left_down + self.dilated_kernel_size - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_down, :, :]*cs_roll[:, :, c, :, :]
                c = mid_left_down - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_left_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_left_down, :, :]*cs_roll[:, :, c, :, :]

                c = mid_right_down + self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_down, :, :]*cs_roll[:, :, c, :, :]
                c = mid_right_down + self.dilated_kernel_size + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_down, :, :]*cs_roll[:, :, c, :, :]
                c = mid_right_down + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid_right_down, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid_right_down, :, :]*cs_roll[:, :, c, :, :]
        else:
            # cs_out[:, :, mid, :, :] = 1

            if self.dilated_kernel_size > 2:
                c = mid + self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + self.dilated_kernel_size + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + -self.dilated_kernel_size + 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + -self.dilated_kernel_size
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + -self.dilated_kernel_size - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]
                c = mid + -1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                c = mid + +self.dilated_kernel_size - 1
                s_out[:, :, c, :, :] = s_roll[:, :, mid, :, :]*s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = cs_roll[:, :, mid, :, :]*cs_roll[:, :, c, :, :]

        for d in range(2, int((self.dilated_kernel_size+1)/2)):

            #corners, next to corners
            # upper left
            corner = mid_left_up - d * self.dilated_kernel_size - d
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_out[:, :, corner+self.dilated_kernel_size+1, :, :].clone()
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_out[:, :, corner+self.dilated_kernel_size+1, :, :].clone()

            c = corner + self.dilated_kernel_size
            s_candidates = s_out[:, :, [c+1, c+self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c+self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * s_roll[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            # upper right
            corner = mid_right_up - d * self.dilated_kernel_size + d
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_out[:, :, corner+self.dilated_kernel_size-1, :, :].clone()
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :] \
                                         * cs_out[:, :, corner+self.dilated_kernel_size-1, :, :].clone()

            c = corner + self.dilated_kernel_size
            s_candidates = s_out[:, :, [c-1, c+self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, c+self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size, c+self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * s_roll[:, :, c, :, :]

            # lower left
            corner = mid_left_down + +d * self.dilated_kernel_size - d
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_out[:, :, corner-self.dilated_kernel_size+1, :, :].clone()
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_out[:, :, corner-self.dilated_kernel_size+1, :, :].clone()

            c = corner - self.dilated_kernel_size
            s_candidates = s_out[:, :, [c+1, c-self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c+1, c-self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            c = corner + 1
            s_candidates = s_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size+1], :, :]
            cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size+1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            # lower right
            corner = mid_right_down + +d * self.dilated_kernel_size + d
            s_out[:, :, corner, :, :] = s_roll[:, :, corner, :, :]\
                                        * s_out[:, :, corner-self.dilated_kernel_size-1, :, :].clone()
            cs_out[:, :, corner, :, :] = cs_roll[:, :, corner, :, :]\
                                         * cs_out[:, :, corner-self.dilated_kernel_size-1, :, :].clone()

            c = corner - self.dilated_kernel_size
            s_candidates = s_out[:, :, [c-1, c-self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-1, -self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            c = corner - 1
            s_candidates = s_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size-1], :, :]
            cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size, c-self.dilated_kernel_size-1], :, :]
            best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (2, 1, 1), return_indices=True)
            s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
            cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]


            # side middles
            steps = range(-(d-2), d-1+((self.dilated_kernel_size +1) % 2))

            # left
            x = -d
            for y in steps:
                c = mid_left_up + y*self.dilated_kernel_size+x
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size+1,
                                            c+1,
                                            c+self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size+1,
                                              c+1,
                                              c+self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

            #top
            y = -d
            for x in steps:
                c = mid_left_up + y*self.dilated_kernel_size+x
                s_candidates = s_out[:, :, [c+self.dilated_kernel_size-1,
                                            c+self.dilated_kernel_size,
                                            c+self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c+self.dilated_kernel_size-1,
                                              c+self.dilated_kernel_size,
                                              c+self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]
            # right
            x = +d
            for y in steps:
                c = mid_right_up + y*self.dilated_kernel_size+x
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size-1,
                                            c-1,
                                            c+self.dilated_kernel_size-1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size-1,
                                              c-1,
                                              c+self.dilated_kernel_size-1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]
            # bottom
            y = +d
            for x in steps:
                c = mid_left_down + y*self.dilated_kernel_size+x
                s_candidates = s_out[:, :, [c-self.dilated_kernel_size-1,
                                            c-self.dilated_kernel_size,
                                            c-self.dilated_kernel_size+1], :, :]
                cs_candidates = cs_out[:, :, [c-self.dilated_kernel_size-1,
                                              c-self.dilated_kernel_size,
                                              c-self.dilated_kernel_size+1], :, :]
                best_candidates, best_inds = torch.nn.functional.max_pool3d(cs_candidates, (3, 1, 1),
                                                                            return_indices=True)
                s_out[:, :, c, :, :] = retrieve_indices(s_candidates, best_inds).squeeze(2) * s_roll[:, :, c, :, :]
                cs_out[:, :, c, :, :] = best_candidates.squeeze(2) * cs_roll[:, :, c, :, :]

        dilation_items = torch.arange(0, self.dilated_kernel_size, self.dilation)
        dilation_items = dilation_items.unsqueeze(0).expand(self.kernel_size, -1).flatten()\
                         + dilation_items.unsqueeze(1).expand(-1, self.kernel_size).flatten()*self.dilated_kernel_size
        s_out = s_out[:, :, dilation_items,
                      : s_out.shape[3]: self.stride, : s_out.shape[4]: self.stride]
        return s_out
