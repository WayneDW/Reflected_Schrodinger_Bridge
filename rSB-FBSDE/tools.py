import numpy as np
from numpy import log, sqrt, sin, cos, exp, pi, prod
from numpy.random import normal, uniform

import torch


class HelperTorch:
    def __init__(self, myClass, device='cpu', max_radius=1, grid_radius=1e-2, grid_curve=1e-3): # finer grid is much slower
        self.myClass = myClass(radius=max_radius)
        self.grid_radius = grid_radius
        self.max_radius = max_radius
        self.cached_points_list = []
        self.device = device
        for radius in np.arange(0., max_radius, max_radius*grid_radius):
            curClass = myClass(radius=radius)
            candidate_points = curClass.position(np.arange(0, 1, grid_curve))
            self.cached_points_list.append(candidate_points)
            
        self.cached_points_list = torch.Tensor(self.cached_points_list).to(self.device)
    
    def inside_domain(self, test_point=torch.Tensor([1, -0.6])):
        test_point = test_point.reshape(1, -1, 1).to(self.device)
        min_rmse = torch.min(torch.sqrt(torch.sum((self.cached_points_list - test_point)**2, dim=1)))
        return min_rmse < self.grid_radius * self.max_radius
    
    def binary_search_boundary(self, left, right):
        if not self.inside_domain(left):
            assert "left should be in the domain."
        if self.inside_domain(right):
            return right

        cnt = 0
        while not self.inside_domain(right) and cnt < 10:
            mid = (left + right) / 2
            if self.inside_domain(mid):
                left = mid.clone()
            else:
                right = mid.clone()
            cnt += 1
        return mid
    
    
    def get_reflection(self, left, right):
        boundary = self.binary_search_boundary(left, right)
        nu = right - boundary
        # compute unit normal vector
        grid_arrays = np.arange(0, 1, self.grid_radius)
        points = torch.Tensor(self.myClass.position(grid_arrays)).to(self.device)
        idx = torch.argmin(torch.sum((points - boundary.reshape(-1, 1))**2, dim=0))
        boundary_t = grid_arrays[idx]
        unit_normal = torch.Tensor(self.myClass.unit_normal(boundary_t)).to(self.device)

        # http://www.sunshine2k.de/articles/coding/vectorreflection/vectorreflection.html
        reflected_nu = nu - 2 * torch.inner(nu, unit_normal) * unit_normal
        reflection_points = boundary + reflected_nu
        return boundary + reflected_nu, boundary


    def map_reflection(self, prev_beta, beta):
        if self.inside_domain(beta):
            return beta
        else:
            reflected_points, boundary = self.get_reflection(prev_beta, beta)
            """ numerical discretization may fail to reflect in extreme cases """
            if not self.inside_domain(reflected_points):
                reflected_points = boundary
            
            return reflected_points