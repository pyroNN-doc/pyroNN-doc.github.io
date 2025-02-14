def init_from_EZRT_header(self,  projection_headers: Tuple[str, ...], reco_header = None ,volume_spacing= None,volume_shape= None,detector_shape = None,detector_spacing = None, swap_detector_axis: bool = True,**kwargs) -> None:
        self.headers = projection_headers
        header = self.headers[0]
        traj_type = 'circ' if np.array_equal(np.array(header.agv_source_position),np.array([0,0,0])) else 'free'
        self.traj_func = arbitrary_projection_matrix
        print(traj_type)
        if traj_type == 'circ':
            angular_range = 2 * np.pi #full circle
            self.parameter_dict['angular_range'] = [0, angular_range]
            
        # if reco_header.num_voxel_z != 0 and reco_header.num_voxel_y != 0 and reco_header.num_voxel_x != 0:
        #     self.parameter_dict['volume_shape'] = np.asarray(
        #         [reco_header.num_voxel_z, reco_header.num_voxel_y, reco_header.num_voxel_x])
        # else:
        #     warnings.warn("Warning: No valid volume shape for Reconstruction could be defined in the geometry")

        # if reco_header.voxel_size_z_in_um != 0 and reco_header.voxel_size_x_in_um != 0:
        #     self.parameter_dict['volume_spacing'] = np.asarray(
        #         [reco_header.voxel_size_z_in_um, reco_header.voxel_size_x_in_um, reco_header.voxel_size_x_in_um]) / 1000.0
        # elif reco_header.voxel_size_in_um != 0:
        #     self.parameter_dict['volume_spacing'] = np.full(shape=3, fill_value=header.voxel_size_in_um / 1000.0,
        #                                                     dtype=self.np_dtype)
        # else:
        #     warnings.warn("Warning: No valid volume spacing for Reconstruction could be defined in the geometry")
        scaling_factor = header._focus_detector_distance_in_um /header._focus_object_distance_in_um
        if reco_header == None:
            if volume_spacing == None:
                # Berechnung der Größen für die Rekonstruktion anhand der Header Daten
                pixel = max([header.number_horizontal_pixels, header.number_vertical_pixels])            
                voxelcount = np.ceil(pixel * scaling_factor*(2/3))
                #Test if voxelcount is dividable by 8
                if voxelcount % 8 != 0:
                    voxelcount = np.ceil(voxelcount/8)*8
                self.parameter_dict['volume_shape'] = np.asarray(
                    [voxelcount,voxelcount,voxelcount])
                self.parameter_dict['volume_spacing'] = np.asarray(
                    [header.detector_width_in_um/pixel,header.detector_width_in_um/pixel,header.detector_width_in_um/pixel]) / 1000.0
            else:
                self.parameter_dict['volume_shape'] = np.asarray(
                    volume_shape)
                self.parameter_dict['volume_spacing'] = np.asarray(
                    [volume_spacing,volume_spacing,volume_spacing])
        elif reco_header.num_voxel_z != 0 and reco_header.num_voxel_y != 0 and reco_header.num_voxel_x != 0:
            self.parameter_dict['volume_shape'] = np.asarray(
                [reco_header.num_voxel_z, reco_header.num_voxel_y, reco_header.num_voxel_x])
            if reco_header.voxel_size_z_in_um != 0 and reco_header.voxel_size_x_in_um != 0:
                self.parameter_dict['volume_spacing'] = np.asarray(
                    [reco_header.voxel_size_z_in_um, reco_header.voxel_size_x_in_um, reco_header.voxel_size_x_in_um]) / (1000.0 /(scaling_factor*1.11))
            elif reco_header.voxel_size_in_um != 0:
                self.parameter_dict['volume_spacing'] = np.full(shape=3, fill_value=header.voxel_size_in_um / 1000.0,
                                                            dtype=self.np_dtype)
        else:
            warnings.warn("Warning: No valid volume shape and/or volume spacing for Reconstruction could be defined in the geometry")

        if self.parameter_dict['volume_shape'] is None or self.parameter_dict['volume_spacing'] is None:
            warnings.warn("Warning: No valid volume origin for Reconstruction could be computed")
        else:
            self.parameter_dict['volume_origin'] = -(self.parameter_dict['volume_shape'] - 1) / 2.0 * \
                                                   self.parameter_dict['volume_spacing']

        # Detector Parameters:
        if detector_shape == None:
            self.parameter_dict['detector_shape'] = np.array(
                [header.number_vertical_pixels, header.number_horizontal_pixels])
            self.parameter_dict['detector_spacing'] = np.full(shape=2, fill_value=(header.detector_height_in_um / 1000.0)/header.number_vertical_pixels,
                                                          dtype=self.np_dtype)  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)
        else:
            self.parameter_dict['detector_shape'] = np.array(detector_shape)
            self.parameter_dict['detector_spacing'] = np.full(shape=2, fill_value=detector_spacing,
                                                          dtype=self.np_dtype)  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)
        
        self.parameter_dict['detector_origin'] = -(self.parameter_dict['detector_shape'] - 1) / 2.0 * \
                                                 self.parameter_dict['detector_spacing']

        # Trajectory Parameters:
        self.parameter_dict['number_of_projections'] = len(projection_headers)#-270#___________________________________________________________________________
        

        self.parameter_dict['sinogram_shape'] = np.array(
            [self.parameter_dict['number_of_projections'], *self.parameter_dict['detector_shape']])
        self.parameter_dict['source_detector_distance'] = header.focus_detector_distance_in_mm
        self.parameter_dict['source_isocenter_distance'] = header.focus_object_distance_in_mm
        self.__generate_trajectory__()

        # Containing the constant part of the distance weight and discretization invarian
        self.parameter_dict['projection_multiplier'] = self.parameter_dict['source_isocenter_distance'] * \
                                                        self.parameter_dict['source_detector_distance'] * \
                                                        self.parameter_dict['detector_spacing'][-1] * np.pi / \
                                                        self.parameter_dict['number_of_projections']
        
        self.parameter_dict['step_size'] = 1.0
        self.parameter_dict['swap_detector_axis'] = swap_detector_axis