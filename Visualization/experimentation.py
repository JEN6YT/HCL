import numpy as np

class Experiment():
    
        ## general experimentation class 
    ## for promotion targeting model 
    def compute_aucc(self, ics, ios): 
        assert(len(ics) == len(ios)) 
        curve_area = 0 
        for i in range(len(ios) - 1): 
            cur_area = 0.5 * (ios[i] + ios[i + 1]) * (ics[i + 1] - ics[i]) 
            curve_area = curve_area + cur_area 
        rectangle_area = ics[-1] * ios[-1] 
        aucc = 1.0 * curve_area / rectangle_area 
        if np.isnan(aucc): 
            import ipdb; ipdb.set_trace() 
        
        return aucc 
    
    def AUC_cpit_cost_curve_deciles_cohort_vis(self, pred_values, values, w, n9d_ni_usd, color, plot_random=False): 
        ## function plots the cost curve the slope of the curve correspond to 
        ## inverse of CPIT as a growth metric 
        ## [TODO] plan to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   n9d_ni_usd: list of next-9-day net inflow in usd per user (cost)        
        
        ValuesControl = values[w < 0.5] 
        NIControl = n9d_ni_usd[w < 0.5] 
        lenControl = len(ValuesControl) 
        
        rpu_control = np.sum(1.0 * ValuesControl) / lenControl 
        
        nipu_control = np.sum(1.0 * NIControl) / lenControl 
        
        print('rpu_control: ' + str(rpu_control)) 
        print('nipu_control: ' + str(nipu_control))
        
        ValuesFT = values[w > 0.5] 
        NIFT = n9d_ni_usd[w > 0.5] 
        lenFT = len(ValuesFT) 
        
        rpu_ft = np.sum(1.0 * ValuesFT) / lenFT

        nipu_ft = np.sum(1.0 * NIFT) / lenFT 

        print('rpu_ft: ' + str(rpu_ft)) 
        print('nipu_ft: ' + str(nipu_ft))
        
        ios = [] 
        ics = [] 
        iopus = [] 
        icpus = [] 
        percs = [] ## let's keep percentages of population, for cross checks 
        cpits = []
        cpitcohorts = []
        treatment_ratios = [] 
        
        ## rewriting the code so that we have an inverse rank 
        pred_values = np.reshape(pred_values, (-1, 1)) 
        indices = np.reshape(range(pred_values.shape[0]), (-1, 1)) 
        
        TD = np.concatenate((pred_values, indices), axis = 1) 
        TD = TD[(-1.0 * TD[:, 0]).argsort(), :] 
        #print('printing out the model scores') 
        #print(TD[0:5])
        ## produce the threshold using a percentage (14%) of users to target 
        #threshold_index = int(0.14 * len(TD)) 
        #threshold = TD[threshold_index, 0] 
        
        """ 
        print('the threshold for choosing 14% of users is: ') 
        print(threshold) 
        print('max: ') 
        print(max(TD[:, 0])) 
        print('min: ') 
        print(min(TD[:, 0])) 
        print('average: ') 
        print(np.mean(TD[:, 0]))
        #exit() 
        """ 
        
        num_segments = 10 
        cnt = 0 
        for idx in range(0, TD.shape[0], int(np.floor(TD.shape[0] / num_segments))): 
            if idx == 0: 
                io = 0 
                ic = 0 
                iopu = 0 
                icpu = 0 
                num_users = 0 
                treatment_ratio = 0 
            else: 
                selected_user_indices = TD[0:idx, 1] 
                
                model_target_users = np.zeros(w.shape) 
                model_target_users[np.reshape(selected_user_indices, (idx,)).astype(np.int)] = 1 ## index 
                
                treated_targeted_filter = np.logical_and(model_target_users, w) 
                untreated_targeted_filter = np.logical_and(model_target_users, (w < 0.5)) 
                treated_untargeted_filter = np.logical_and(model_target_users < 0.5, w ) 
                untreated_untargeted_filter = np.logical_and(model_target_users < 0.5, (w < 0.5)) 
                
                perc = sum(1.0 * model_target_users) / len(model_target_users) 
                
                treated_target_rpu = sum(1.0 * values[treated_targeted_filter]) / sum(treated_targeted_filter) 
                treated_target_nipu = sum(1.0 * n9d_ni_usd[treated_targeted_filter]) / sum(treated_targeted_filter) 
                
                untreated_target_rpu = sum(1.0 * values[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                untreated_target_nipu = sum(1.0 * n9d_ni_usd[untreated_targeted_filter]) / sum(untreated_targeted_filter) 
                
                treated_untarget_rpu = sum(1.0 * values[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                treated_untarget_nipu = sum(1.0 * n9d_ni_usd[treated_untargeted_filter]) / sum(treated_untargeted_filter) 
                
                untreated_untarget_rpu = sum(1.0 * values[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                untreated_untarget_nipu = sum(1.0 * n9d_ni_usd[untreated_untargeted_filter]) / sum(untreated_untargeted_filter) 
                
                if perc > 0.99: 
                    print('rpu_cohort') 
                    print(rpu_cohort) 
                    print('treated_target_rpu') 
                    print(treated_target_rpu) 
                    rpu_cohort = treated_target_rpu 
                    nipu_cohort = treated_target_nipu 
                else: 
                    rpu_cohort = treated_target_rpu * perc + untreated_untarget_rpu * (1 - perc) 
                    nipu_cohort = treated_target_nipu * perc + untreated_untarget_nipu * (1 - perc) 
                
                iopu = max(rpu_cohort - rpu_control, 1e-7) 
                icpu = max(-1.0 * (nipu_cohort - nipu_control), 1e-7) 
                
                io = iopu * lenControl 
                ic = icpu * lenControl 
                
                cpit = icpu / iopu 
                
                #cpitpc = -1.0 * (treated_target_nipu - nipu_control) / (treated_target_rpu - rpu_control) 
                cpitcohort = -1.0 * (nipu_cohort - nipu_control) / (rpu_cohort - rpu_control) 

                percs.append(perc)
                cpits.append(cpit)
                cpitcohorts.append(cpitcohort)
                
                liftvscontrol = (rpu_cohort - rpu_control) * 1.0 / (rpu_control) 
                liftrandomvscontrol = (rpu_ft - rpu_control) * 1.0 / (rpu_control) 
                liftpcvscontrol = (treated_target_rpu - rpu_control) * 1.0 / (rpu_control) 
                
                if True: #perc > 0.35 and perc < 0.45: 
                    print('---------------------------->>>>>>') 
                    print('perc - target: %.2f' % perc) 
                    print('treated_target_rpu: %.2f' % treated_target_rpu) 
                    print('treated_target_nipu: %.2f' %treated_target_nipu) 
                    print('nontreated_target_rpu: %.2f' % untreated_target_rpu) 
                    print('nontreated_target_nipu: %.2f' % untreated_target_nipu) 
                    print('treated_nontarget_rpu: %.2f' %treated_untarget_rpu) 
                    print('treated_nontarget_nipu: %.2f' %treated_untarget_nipu) 
                    print('nontreated_nontarget_rpu: %.2f' %untreated_untarget_rpu) 
                    print('nontreated_nontarget_nipu: %.2f' %untreated_untarget_nipu) 
                    
                    cpit_targeted = -1.0 * (treated_target_nipu - untreated_target_nipu) / (treated_target_rpu - untreated_target_rpu) 
                    cpit_nontargeted = -1.0 * (treated_untarget_nipu - untreated_untarget_nipu) / (treated_untarget_rpu - untreated_untarget_rpu) 
                    
                    print('--- with ' + str(perc * 100) + '% targeting, print cpits to treat users and create incrementality in users ---') 
                    print('--> in targeted users: ') 
                    print('cpit = ' + str(cpit_targeted)) 
                    print('--> in non-targeted users: ') 
                    print('cpit = ' + str(cpit_nontargeted)) 
                    
                    print('rpu_control: %.2f' %rpu_control) 
                    print('nipu_control: %.2f' %nipu_control) 
                    print('rpu_ft: %.2f' %rpu_ft) 
                    print('nipu_ft: %.2f' %nipu_ft) 
                    print('rpu_cohort: %.2f' %rpu_cohort) 
                    print('nipu_cohort: %.2f' %nipu_cohort) 
                    
                    print('lift targeted cohort vs control: %.2f' %liftvscontrol) 
                    print('lift random vs control: %.2f' %liftrandomvscontrol) 
                    print('cpit cohort vs control: %.2f' %cpit) 
                    print('lift targeted-treated vs control: %.2f' %liftpcvscontrol) 
                    print('cpit cohort: %2f' %cpitcohort) 
            
            ics.append(ic) 
            ### ---- guard against the cost curve going down --- 
            ### for plotting and visualization 
            if len(ios) > 0: 
                if io < ios[-1]: 
                    io = ios[-1] 
            ios.append(io) 
            
            icpus.append(icpu) 
            iopus.append(iopu) 
            treatment_ratios.append(treatment_ratio) 
            
            """ 
            print('decile: ' + str(cnt)) 
            print('cost: ' + str(ic)) 
            print('increment orders: ' + str(io)) 
            print('increment orders per user: ' + str(iopu)) 
            if io != 0: 
                print('cpit: ' + str(ic * 1.0 / io)) 
            """ 
            cnt = cnt + 1 
        
        ## sort the tuple by its index (cost) 
        ics = np.asarray(ics); ics = np.reshape(ics, (-1, 1)) 
        ios = np.asarray(ios); ios = np.reshape(ios, (-1, 1)) 
        ips = np.minimum(ics / ics[-1], 1.0) 
        ios = np.minimum(ios / ios[-1], 1.0) 
        
        if plot_random == True: 
            ips = np.reshape(np.arange(0.0, 1.1, 0.1), (-1, 1)) 
            ios = np.reshape(np.arange(0.0, 1.1, 0.1), (-1, 1)) 
        
        ### to guarantee to remove nan issues 
        ips[-1] = 1.0 
        ios[-1] = 1.0 
        
        #ips = ics 
        #ios = ios 
        combined_series = np.concatenate((ics, ios), axis=1) 
        combined_series = np.concatenate((combined_series, ips), axis=1) 
        
        #combined_series = combined_series[combined_series[:, 0].argsort(), :] 
        ## plotting on different colors on same figure 
        ## -- combined_series [:, 0] contains cost 
        ## -- combined_series [:, 1] contains trips/orders 
        ## -- combined_series [:, 2] contains percentages 
        
        ### [for Hong:] feel free to save this to file 
        ### let's define the interface between eng and vis 
        # plt.plot(combined_series[:, 2], combined_series[:, 1], '-o'+color, markersize=12, linewidth=3)
        ### [Todo:] define the file format
        ### ask Bhavya about this evaluator interface
        
        aucc = self.compute_aucc(ips, ios) 
        # plt.xlabel('Incremental cost % of maximum')
        # plt.ylabel('Incremental value % of maximum')

        # ax = plt.gca()
        # type(ax)  # matplotlib.axes._subplots.AxesSubplot

        # # manipulate to use percentage
        # vals = ax.get_xticks()
        # #ax.set_xticklabels(['{:,.1%}'.format(x) for x in vals])
        # vals = ax.get_yticks()
        # #ax.set_yticklabels(['{:,.1%}'.format(x) for x in vals])
        # plt.grid(True)
        return aucc, percs, cpits, cpitcohorts
    
    
    def compute_filters_AtBt(self, w, model_target_users): 
        
        # compute filters for At and Bt sets in our documentation 
        # https://docs.google.com/document/d/1c8uCtvh71_Px1PS4NfdF0qDpA-8rZ10GtFYUaGkc6F4/edit 
        # these are sets where 
        #   At: a user is 'selected/targeted' and 'treated' 
        #   Bt: a user is 'selected/targeted' and 'not-treated' 
        # 
        # w: treatment label 
        # model_target_users: the binary vector indicating whether each user is targeted 
        
        target_treated_users = np.logical_and((w > 0.5), (model_target_users > 0.5)) 
        target_nontreated_users = np.logical_and((w <= 0.5), (model_target_users > 0.5)) 
        return target_treated_users, target_nontreated_users 

    def compute_increment_orders_target(self, values, w, model_target_users, n9d_ni_usd): 
            
        ## compute incremental orders and incremental cost series 
        ## then plot them to construct a cost curve for selection/targeting 
        ## model evaluation 
        ## values: the value representation or # of orders 
        ## w: the treatment labels 
        ## model_target_users: binary vector indicating whether the user is targeted per dimension 
        ## n9d_ni_usd: the next 9 days inflow numbers for evaluation of incremental cost 
        
        target_treated_users, target_nontreated_users = self.compute_filters_AtBt(w, model_target_users) 
        
        num_users = np.sum(model_target_users) 
        
        values_At = values[target_treated_users] 
        values_Bt = values[target_nontreated_users] 
        
        #values_At = np.minimum(values_At, 20.0) 
        #values_Bt = np.minimum(values_Bt, 20.0) 
        
        nis_At = n9d_ni_usd[target_treated_users] 
        nis_Bt = n9d_ni_usd[target_nontreated_users] 
        
        try: 
            a = 1.0 * np.sum(values_At) 
            b = 1.0 * np.sum(values_Bt) 
            io = a - b 
            
            c = 1.0 * np.sum(nis_At) 
            d = 1.0 * np.sum(nis_Bt) 
            
            # incremental cost: negative of difference in net-inflow 
            ic = -1.0 * ( c - d ) 
            
            # handle divide by zeros 
            a = a / len(values_At) 
            b = b / len(values_Bt) 
            c = c / len(nis_At) 
            d = d / len(nis_Bt) 
            
            ## incremental order per user and its percentage version 
            oput = a 
            opun = b 
            cput = c 
            cpun = d 
            iopu =  a - b 
            icpu =  -1.0 * ( c - d ) 
            
            #num_treated = len(values_At) 
            treatment_ratio = 1.0 * len(values_At) / (len(values_At) + len(values_Bt)) 
            
            num_treated = len(values_At) 
            
            io = iopu * num_treated 
            ic = icpu * num_treated
            
            #io = oput * num_users * treatment_ratio - opun * num_users * (1.0 - treatment_ratio) 
            
            #io = 0.5 * num_users * iopu  
            #ic = 0.5 * num_users * icpu 
        except: 
            io = 0 
            ic = 0 
            iopu = 0 
            icpu = 0 
            treatment_ratio = 0.0 
        
        return io, ic, iopu, icpu, num_users, treatment_ratio

    def AUC_ivpu(self, pred_values, values, w, thresh_list, color, n9d_ni_usd, plot=False):
        ## function plots ivpu versus population percentage 
        ## and [TODO] to compute Area Under Curve (AUC) 
        ##   pred_values: list of model predictions, same vertical dim as # data points 
        ##   values: actual labels (reward or value) 
        ##   w: treatment labels {1,0} 
        ##   thresh_list: a range of thresholds generated by numpy.arange based on histogram of pred_values 
        
        percs = [] 
        iopus = [] 
        icpus = [] 
        for t in thresh_list: 
            model_target_users = pred_values > t 
            try: # handle divide by zero 
                perc_users = 1.0 * np.sum(model_target_users) / len(model_target_users) 
            except: 
                perc_users = 0.0 
            d1, d2, iopu, icpu, num_users, treatment_ratio = self.compute_increment_orders_target(values, w, model_target_users, n9d_ni_usd) 
            percs.append(perc_users) 
            iopus.append(iopu) 
            icpus.append(icpu)
        
        # if plot:
        #     ## plotting
        #     plt.plot(percs, iopus, '-o'+color, markersize=12, linewidth=3)
        #     plt.xlabel('% population covered')
        #     plt.ylabel('Incremental # of orders per user')
        #     #plt.ylim([0, 0.3])
        #     plt.grid(True)
        
        return percs, iopus, icpus