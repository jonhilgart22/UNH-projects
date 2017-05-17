#! usr/bin/env python
# -*- coding: UTF-8 -*-
import simpy
import scipy.stats as stats
import pandas as pd
import numpy as np
import time
import pickle
import sys
from esp_product_revenue import ESP_revenue_predictions
from ESP_Markov_Model_Client_Lifetime import ESP_Joint_Product_Probabilities, \
    ESP_Markov_Model_Joint_Prob
__author__ = 'Jonathan Hilgart'


class Client(object):
    """This is the base class to represent client that enter the bank.
    The class will contain attributes such as lifetime, when their bank account
    was opened, simpy requests for different resources .

    A general class to represent all of the atributes of a client.
    Used in the Markov MOdel to make product inferences. """

    def __init__(self, client_id):
        """Initialize with the client ID and lifetime. As the time progresses,
        keep track of different events that occur."""
        self.client_id = client_id
        self.client_lifetime = None
        self.time_bank_closed = None
        self.time_credit_card_opened = None
        self.time_bank_was_opened = None
        # Store the resource requests for SImpy
        self.esp_open_money_market_bonus_request = None
        self.esp_open_collateral_mma_request  = None
        self.esp_open_cash_management_request = None
        self.esp_open_fx_request = None
        self.esp_open_letters_of_credit_request = None
        self.esp_open_enterprise_sweep_request = None
        self.esp_open_checking_request = None

        self.esp_client_alive_resource_request = None


        self.have_mmb = 0
        self.have_cmma = 0
        self.have_cm = 0
        self.have_fx = 0
        self.have_loc = 0
        self.have_es = 0
        self.have_checking = 0

        self.client_age = None

        self.close_account_process = None




class ESP_flow(object):
    """Model cclients in ESP opening up produts over time.
    Client lifetime drawn from distribution of client lifetimes from 2013-2016.
    The probability of ech product is inferred from a Markov Model, where the
    factors between the product nodes represent the joint probabilities. These
    join probabilities are updated are every week number to performance inference.

    The revenue per product is drawn from 2016 historical data (per month)
    The number of clients per week is drawn from 2016 data.

    Note, all time units are in terms of one week. One day would correspond
    to 1/7 of a week or .143."""
    def __init__(self, env, number_of_weeks_to_run,
                 yearly_interest_rate,
                 increase_esp_clients_percent_per_week =0,
                 esp_client_alive_resource = 10000,
                 esp_open_money_market_bonus_capacity=5000,
                 esp_open_collateral_mma_capacity =5000,
                 esp_open_cash_management_capacity = 5000, esp_fx_capacity = 5000,
                 esp_open_letters_of_credit_capacity = 5000,
                 esp_open_enterprise_sweep_capacity = 5000, esp_open_checking_capacity = 5000,
                 cc_capacity=200, esp_capacity = 5000,
                 stripe_capacity=3000,
                 evidence_ = None):
        self.env = env
        self.list_of_all_clients = []
        self.weekly_interest_rate = self.yearly_to_weekly_interest_rate_conversion(
            yearly_interest_rate)

        self.number_of_weeks_to_run = number_of_weeks_to_run

        self.esp_money_market_bonus_resource = simpy.Resource(env, capacity=esp_open_money_market_bonus_capacity)
        self.esp_collateral_mma_resource = simpy.Resource(env, capacity=esp_open_collateral_mma_capacity)
        self.esp_cash_management_resource=  simpy.Resource(env, capacity=esp_open_cash_management_capacity)
        self.esp_fx_resource = simpy.Resource(env, capacity=esp_fx_capacity)
        self.esp_letters_of_credit_resource = simpy.Resource(env, capacity=esp_open_letters_of_credit_capacity )
        self.esp_enterprise_sweep_resource = simpy.Resource(env, capacity=esp_open_enterprise_sweep_capacity )
        self.esp_checking_resource = simpy.Resource(env, capacity= esp_open_checking_capacity )
        self.esp_client_alive_resource = simpy.Resource(env, capacity= esp_client_alive_resource )

        self.time_series_total_clients = []
        self.time_series_cumulative_clients = []

        self.time_series_esp_money_market_bonus = []
        self.time_series_esp_collateral_mma = []
        self.time_series_esp_cash_management = []
        self.time_series_esp_fx = []
        self.time_series_esp_letters_of_credit = []
        self.time_series_esp_enterprise_sweep = []
        self.time_series_esp_checking= []

        self.time_series_esp_money_market_bonus_total_weekly_rev= []
        self.time_series_esp_collateral_mma_total_weekly_rev = []
        self.time_series_esp_cash_management_total_weekly_rev= []
        self.time_series_esp_fx_total_weekly_rev= []
        self.time_series_esp_letters_of_credit_total_weekly_rev = []
        self.time_series_esp_enterprise_sweep_total_weekly_rev= []
        self.time_series_esp_checking_total_weekly_rev = []

        self.time_series_esp_money_market_bonus_rev_per_customer = []
        self.time_series_esp_collateral_mma_rev_per_customer = []
        self.time_series_esp_cash_management_rev_per_customer = []
        self.time_series_esp_fx_rev_per_customer = []
        self.time_series_esp_letters_of_credit_rev_per_customer = []
        self.time_series_esp_enterprise_sweep_rev_per_customer = []
        self.time_series_esp_checking_rev_per_customer = []
        # If we wanted to simulate increasing the number of new ESP customers per week
        self.increase_esp_clients_percent_per_week  = \
            increase_esp_clients_percent_per_week

        # Store any initial evidence we have for products
        self.evidence = evidence_

    def yearly_to_weekly_interest_rate_conversion(self,yearly_interest_rate):
        """Convert from a yearly rate to a weekly rate using the following
        equation.
        Effective rate for period = (1 + annual rate)**(1 / # of periods) – 1
        """
        weekly_interest_rate  = ((1 + yearly_interest_rate)**(1/52))-1
        return weekly_interest_rate

    def esp_clients_per_week(self,mean=20.433962264150942, std=3.5432472792051746):
        """This generates the number of new clients in ESP for a given week.
        The default parameters are taken from the years 2013-2016."""
        self.oneweek_esp_clients = round(stats.norm.rvs(mean,std))
        if self.oneweek_esp_clients <0:
            self.oneweek_esp_clients = 0
        self.oneweek_esp_clients = self.oneweek_esp_clients * \
            self.increase_esp_clients_percent_per_week + self.oneweek_esp_clients

    def accelerator_clients_per_week(self,mean=4.1792452830188678,
                                     std=0.92716914151900442):
        """This generates the number of new clients in accelerator for a given week.
        The default parameters are taken from the years 2013-2016"""
        self.oneweek_accelerator_clients = round(stats.norm.rvs(mean,std))
        if self.oneweek_accelerator_clients < 0:
            self.oneweek_accelerator_clients =0

    def stripe_clients_per_week(self,mean =23.209302325581394,
                                std =12.505920717868896):
        """"This generates the number of new Stripe customers from the given week.
        The default parameters from from 2016"""

        self.oneweek_stripe_clients = round(stats.norm.rvs(mean, std))
        if self.oneweek_stripe_clients < 0:
            self.oneweek_stripe_clients = 0

    def time_between_esb_accelerator(self,shape = 1.3513865965152867,
        location = -0.85750795314579964, scale = 57.412494398862549):
        """This is an exponential distribution of the average time between
        a client being in the esp team and being moved to the acceleartor team.
        Default parameters are from 2000-2016"""
        self.time_between_esb_accelerator = stats.gamma.rvs(shape, location, scale)
        if self.time_between_esb_accelerator <0:
            self.time_between_esb_accelerator = 1
            # at least one week before transferring to accelerator

    def esp_client_lifetime(self):
        """Draws from a distribution of client lifetimes (in months) from 2013-2016.
        Return the number of weeks that a client will be alive.

        A client needs to be generating revenue for at least three months, and not
        have generated revenue for three months to be considred a
        'client lifetime'. It is possible for a single client to have Multiple
        'client lifetimes' that feed into the parameters for the Exponential
        distribution.

        Multiply the result by 4 to turn months into weeks"""
        exponential_lifetime_parameters = (2.9999999999982676, 11.500665661185888)
        return round(stats.expon(*exponential_lifetime_parameters ).rvs())*4


    def initiate_week_client_run(self, esp_mean=20.433962264150942,
        esp_std=3.5432472792051746, accel_mean = 4.1792452830188678,
        accel_std = 0.92716914151900442, stripe_mean = 23.209302325581394,
        stripe_std = 12.505920717868896):
        """This is the main function for initiating clients throughout the time
        at the bank. The number of customers is an input which is given when you
        instantiate this class.

        The esp, accelerator  and stripe mean come from 2000-2016

        This function steps through the simulated time one week at a time
        and keeps track of the number of clients at each node during this simulation.

        This function looks at the probabilities associated with each product
        via a dynamic Markov Model (where the edges represent joint probabilities
        between products that are updated every week). These probabilities are used
        to infer the probability of a client having each product given evidence
        about the other products that client has.

        This function keeps track of the total number of clients over time,
        the total number of clients for the seven ESP products, the total
        GP per week per product, and the GP per client per product (adjusted
        to be in NPV).

        In addition, each client lifetime, which is drawn from an exponential
        distribution, is represented by a simpy process. Once a client churns,
        they are no longer counted in each of the products that used to hold.


        """
        for week_n in range(self.number_of_weeks_to_run):
            print("Starting WEEK NUMBER {}".format(week_n))
            # generate new clients for each channel
            self.esp_clients_per_week(esp_mean, esp_std)
            self.accelerator_clients_per_week(accel_mean, accel_std)
            self.stripe_clients_per_week(stripe_mean, stripe_std)
            print(self.oneweek_esp_clients, ' ESP clients this week')
            # Keep track of the total number of clients over time
            self.time_series_total_clients.append(
                ('Total New clients IN ESP for Week = ', week_n, self.oneweek_esp_clients))
            # Total number of clients



            ## See where the ESP clients end up across the products
            for esp_client_n in range(int(self.oneweek_esp_clients)):

                # Client id is number for the week + week number
                esp_client = Client(str(esp_client_n)+'-'+str(week_n))
                 # default client lifetime
                esp_client.client_age = 0 # new client
                # keep track of the total number of clients over time
                client_alive = self.esp_client_alive_resource.request()
                yield client_alive
                esp_client.esp_client_alive_resource_request = client_alive



                # Draw a lifetime value from an exponential distribution
                esp_client.client_lifetime = self.esp_client_lifetime()

                # make a list of all esp clients
                self.list_of_all_clients.append(esp_client)

                # keep track of cumulative clients
                self.time_series_cumulative_clients.append(("ESP cumulative clients\
                     for week =", week_n, self.esp_client_alive_resource.count))

            for idx,client in enumerate(self.list_of_all_clients):
                # print client lifetime (see when clients are closing accounts)
                if idx % 10 == 0:
                    print('Client {} lifetime = {}'.format(client.client_id,
                                                           client.client_lifetime))



                # Yield for the client lifetime
                # only span one close account process per client
                # Otherwise, SImpy will try to close the same account
                # Multiple times
                if client.close_account_process == None:
                    close_accounts = self.env.process(self.close_accounts(client))
                    client.close_account_process = close_accounts

                if client.client_age == 0: ## Don't have any products yet
                    checking_prob, cmma_prob, mmb_prob, cm_prob, fx_prob ,loc_prob, es_prob = \
                    ESP_Markov_Model_Joint_Prob(ESP_Joint_Product_Probabilities,
                            single=True,week_n_one_time=client.client_age,
                            evidence_ = self.evidence)
                    # See if a client has each product
                    open_checking = np.random.choice([1,0],p=np.array(
                        [checking_prob,(1-checking_prob)]))
                    open_cmma = np.random.choice([1,0],p=np.array(
                        [cmma_prob,(1-cmma_prob)]))
                    open_mmb = np.random.choice([1,0],p=np.array(
                        [mmb_prob,(1-mmb_prob)]))
                    open_cm = np.random.choice([1,0],p=np.array(
                        [cm_prob,(1-cm_prob)]))
                    open_fx = np.random.choice([1,0],p=np.array(
                        [fx_prob,(1-fx_prob)]))
                    open_loc = np.random.choice([1,0],p=np.array(
                        [loc_prob,(1-loc_prob)]))
                    open_es = np.random.choice([1,0],p=np.array(
                        [es_prob,(1-es_prob)]))


                    # open an account if a client has each product
                    # Otherwise, add a default event to yield
                    if open_checking == 1:
                        if client.have_checking == 0:
                            open_checking = self.env.process(self.esp_open_checking(client))
                        else:
                            open_checking = self.env.timeout(0)
                        # either open product or
                    else:
                        open_checking = self.env.timeout(0)

                    if open_cmma == 1:
                        if client.have_cmma == 0:
                            open_cmma = self.env.process(
                                self.esp_open_collateral_mma(client))
                        else:
                            open_cmma = self.env.timeout(0)
                        # yield close_accounts |open_cmma
                    else:
                        open_cmma = self.env.timeout(0)

                    if open_mmb ==1:
                        if client.have_mmb == 0:
                            open_mmb = self.env.process(
                                self.esp_open_money_market_bonus(client))
                        else:
                            open_mmb = self.env.timeout(0)
                    else:
                        open_mmb = self.env.timeout(0)
                    if open_cm == 1:
                        if client.have_cm == 0:
                            open_cm = self.env.process(
                                self.esp_open_cash_management(client))
                        else:
                            open_cm = self.env.timeout(0)
                        # yield close_accounts | open_cm
                    else:
                        open_cm = self.env.timeout(0)
                    if open_fx == 1:
                        if client.have_fx == 0:
                            open_fx = self.env.process(self.esp_open_fx(client))
                        else:
                            open_fx = self.env.timeout(0)
                        # yield close_accounts |open_fx
                    else:
                        open_fx = self.env.timeout(0)
                    if open_loc == 1:
                        if client.have_fx == 0:

                            open_loc = self.env.process(
                                self.esp_open_letters_of_credit(client))
                        else:
                            open_loc = self.env.timeout(0)
                        # yield close_accounts | open_loc
                    else:
                        open_loc = self.env.timeout(0)
                    if open_es == 1:
                        if client.have_es == 0:
                            open_es = self.env.process(
                                self.esp_open_enterprise_sweep(client))
                        else:
                            open_es = self.env.timeout(0)
                        # yield close_accounts | open_es
                    else:
                        open_es = self.env.timeout(0)
                    # either open product or close the account
                    yield (open_checking &open_cmma & open_mmb & open_cm \
                           &open_fx & open_loc & open_es) | client.close_account_process


                    client.client_age +=1  # increment the age of the client

                else:
                    # every client now has an indicator for if they have
                    # a product or not

                    checking_prob, cmma_prob, mmb_prob, cm_prob, fx_prob ,loc_prob, es_prob = \
                    ESP_Markov_Model_Joint_Prob(ESP_Joint_Product_Probabilities,
                            single=True,week_n_one_time=client.client_age,
                            evidence_={'money_market_bonus':client.have_mmb,
                                       'collateral_mma':client.have_cmma,
                                'cash_management':client.have_cm,
                                'enterprise_sweep':client.have_es,
                                'fx_products':client.have_fx,
                                'letters_of_credit':client.have_loc,
                                'checking_usd':client.have_checking})
                    # # update if these clients have each product


                    ## See if a client has each product
                    open_checking = np.random.choice([1,0],p=np.array(
                        [checking_prob,(1-checking_prob)]))
                    open_cmma = np.random.choice([1,0],p=np.array(
                        [cmma_prob,(1-cmma_prob)]))
                    open_mmb = np.random.choice([1,0],p=np.array(
                        [mmb_prob,(1-mmb_prob)]))
                    open_cm = np.random.choice([1,0],p=np.array(
                        [cm_prob,(1-cm_prob)]))
                    open_fx = np.random.choice([1,0],p=np.array(
                        [fx_prob,(1-fx_prob)]))
                    open_loc = np.random.choice([1,0],p=np.array(
                        [loc_prob,(1-loc_prob)]))
                    open_es = np.random.choice([1,0],p=np.array(
                        [es_prob,(1-es_prob)]))

                    # open an account if a client has each product
                    # Otherwise, add a default event to yield
                    if open_checking == 1:
                        if client.have_checking == 0:
                            open_checking = self.env.process(self.esp_open_checking(client))
                        else:
                            open_checking = self.env.timeout(0)
                        # either open product or
                    elif open_checking == 0:
                        if client.have_checking ==1 :
                            open_checking = self.env.process(self.esp_close_checking(client))
                        else:
                            open_checking = self.env.timeout(0)
                    else:
                        print('Something weird happened')

                        # close the account for this client product
                        open_checking = self.env.timeout(0)

                    if open_cmma == 1:
                        if client.have_cmma == 0:
                            open_cmma = self.env.process(
                                self.esp_open_collateral_mma(client))
                        else:
                            open_cmma = self.env.timeout(0)
                        # yield close_accounts |open_cmma
                    elif open_cmma == 0 :
                        if client.have_cmma ==1:
                            open_cmma =  self.env.process(self.esp_close_collateral_mma(client))
                        else:
                            open_cmma = self.env.timeout(0)
                    else:
                        print('Something weird happened')

                    if open_mmb ==1:
                        if client.have_mmb == 0:
                            open_mmb = self.env.process(
                                self.esp_open_money_market_bonus(client))
                        else:
                            open_mmb = self.env.timeout(0)
                    elif open_mmb == 0:
                        if client.have_mmb == 1:
                            open_mmb = self.env.process(self.esp_close_money_market_bonus(client))
                        else:
                            open_mmb = self.env.timeout(0)
                    else:
                        pass


                    if open_cm == 1:
                        if client.have_cm == 0:
                            open_cm = self.env.process(
                                self.esp_open_cash_management(client))
                        else:
                            open_cm = self.env.timeout(0)
                        # yield close_accounts | open_cm
                    elif open_cm == 0:
                        if client.have_cm == 1:
                            open_cm =  self.env.process(self.esp_close_cash_management(client))
                        else:
                            open_cm = self.env.timeout(0)
                    else:
                        pass


                    if open_fx == 1:
                        if client.have_fx == 0:
                            open_fx = self.env.process(self.esp_open_fx(client))
                        else:
                            open_fx = self.env.timeout(0)
                        # yield close_accounts |open_fx
                    elif open_fx ==0:
                        if client.have_fx == 1:
                            open_fx = self.env.process(self.esp_close_fx(client))
                        else:
                            open_fx = self.env.timeout(0)
                    else:
                        pass


                    if open_loc == 1:
                        if client.have_loc == 0:

                            open_loc = self.env.process(
                                self.esp_open_letters_of_credit(client))
                        else:
                            open_loc = self.env.timeout(0)
                        # yield close_accounts | open_loc
                    elif open_loc == 0:
                        if client.have_loc == 1:
                            open_loc = self.env.process(self.esp_close_letters_of_credit(client))
                        else:
                            open_loc = self.env.timeout(0)
                    else:
                        pass


                    if open_es == 1:
                        if client.have_es == 0:
                            open_es = self.env.process(
                                self.esp_open_enterprise_sweep(client))
                        else:
                            open_es = self.env.timeout(0)
                        # yield close_accounts | open_es
                    elif open_es == 0:
                        if client.have_es == 1:
                            open_es =  self.env.process(self.esp_close_enterprise_sweep(client))
                        else:
                            open_es = self.env.timeout(0)

                    else:
                        pass
                    # either open product or close the account
                    yield (open_checking & open_cmma & open_mmb & open_cm \
                           &open_fx & open_loc & open_es) | client.close_account_process

                    client.client_age +=1 # increment the age of the client
                if idx % 10 == 0 :
                    ## print out stats of every 10th client
                    print(client.client_id, ' client id ')
                    print(client.client_age,'client age')
                    print(client.have_mmb, ' client.have_mmb')
                    print(client.have_cmma, 'client.have_cmma')
                    print(client.have_cm, 'client.have_cm')
                    print( client.have_es, ' client.have_es')
                    print(client.have_fx,'client.have_fx')
                    print(client.have_loc,'client.have_loc')
                    print(client.have_checking,'client.have_checking')


            # print the weekly metrics
            print()
            print('WEEK METRICS {}'.format(week_n))
            print(self.esp_money_market_bonus_resource.count,'esp mmb clients ')
            print(self.esp_collateral_mma_resource.count, ' esp cmma clients')
            print(self.esp_cash_management_resource.count, ' esp cm clients')
            print(self.esp_fx_resource.count, 'fx count')
            print(self.esp_letters_of_credit_resource.count, ' loc count')
            print(self.esp_enterprise_sweep_resource.count, 'es count')
            print(self.esp_checking_resource.count , 'checking count')
            print(self.esp_client_alive_resource.count, ' total number of clients')
            print()


            # At the end of each week, record the number of clients per
            # product
            self.time_series_esp_money_market_bonus.append(("Week = ",
                self.env.now,self.esp_money_market_bonus_resource.count))
            self.time_series_esp_collateral_mma.append(("Week = ",
                self.env.now,self.esp_collateral_mma_resource.count))
            self.time_series_esp_cash_management.append(("Week = ",
                self.env.now,self.esp_cash_management_resource.count))
            self.time_series_esp_fx.append(("Week = ",
                self.env.now,self.esp_fx_resource.count))
            self.time_series_esp_letters_of_credit.append(("Week = ",
                self.env.now,self.esp_letters_of_credit_resource.count))
            self.time_series_esp_enterprise_sweep.append(("Week = ",
                self.env.now,self.esp_enterprise_sweep_resource.count))
            self.time_series_esp_checking.append(("Week = ",
                self.env.now,self.esp_checking_resource.count))
            # At the end of each week, find the weekly GP and weekly GP per client
            # esp money market bonus weekly gp

            self.get_weekly_gp(week_n, self.time_series_esp_money_market_bonus,
                ESP_revenue_predictions,
                self.time_series_esp_money_market_bonus_total_weekly_rev,
                self.time_series_esp_money_market_bonus_rev_per_customer,
                'mmb')
            # esp collateral mma weekly gp

            self.get_weekly_gp(week_n, self.time_series_esp_collateral_mma,
                ESP_revenue_predictions,
                self.time_series_esp_collateral_mma_total_weekly_rev,
                self.time_series_esp_collateral_mma_rev_per_customer,
                'cmma')
            # esp cash management weekly revenue
            self.get_weekly_gp(week_n, self.time_series_esp_cash_management,
                ESP_revenue_predictions,
                self.time_series_esp_cash_management_total_weekly_rev,
                self.time_series_esp_cash_management_rev_per_customer,
                'cm')
            ### esp fx weekly gp
            self.get_weekly_gp(week_n, self.time_series_esp_fx,
                ESP_revenue_predictions,
                self.time_series_esp_fx_total_weekly_rev,
                self.time_series_esp_fx_rev_per_customer,
                'fx')
            ### esp letters of credit
            self.get_weekly_gp(week_n, self.time_series_esp_letters_of_credit,
                ESP_revenue_predictions,
                self.time_series_esp_letters_of_credit_total_weekly_rev,
                self.time_series_esp_letters_of_credit_rev_per_customer,
                'loc')
            ### esp enterprise sweep weekly gp
            self.get_weekly_gp(week_n, self.time_series_esp_enterprise_sweep,
                ESP_revenue_predictions,
                self.time_series_esp_enterprise_sweep_total_weekly_rev,
                self.time_series_esp_enterprise_sweep_rev_per_customer,
                'es')
            ### esp checking weekly gp
            self.get_weekly_gp(week_n, self.time_series_esp_checking,
                ESP_revenue_predictions,
                self.time_series_esp_checking_total_weekly_rev,
                self.time_series_esp_checking_rev_per_customer,
                'checking')

            # Increment by one week
            one_week_increment = self.env.timeout(1)
            yield one_week_increment



    def monitor_resource(self, resource, resource_name):
        """Print out monitoring statistics for a given resource.
        NUmber of slots allocated.
        Number of people using the resource
        Number of queued events for the resource"""
        print()
        print("MONITORING STATISTICS FOR {}".format(resource_name))
        print('{} of {} slots are allocated at time {}.'.format (
            resource.count, resource.capacity, self.env.now))
        #print('  Users :', resource.users)
        print('  Queued events:', resource.queue)
        print()

    def get_weekly_gp(self,week_n, time_series, gp_data_function, total_rev_week,
                      rev_per_client_week, product):
        """Get the total revenue for the week, and revenue per client, for a
        given product.
        Also, adjusts the revenue to be in week zero through net
        present value.
        Need the weekly return interest rate.

        NPV = ∑ {Net Period Cash Flow / (1+R)^T} - Initial Investment """
        total_weekly_rev = 0
        number_of_customer_for_product_week = time_series[week_n][2]
        if number_of_customer_for_product_week == 0:
            # No customer this week for this product
            total_rev_week.append( ('week = ',week_n, 0))
            rev_per_client_week.append( ('week = ',week_n,0))

        else: # We have customers!

            for esp_customer in range(number_of_customer_for_product_week ):
                # Get weekly revenue from distribution
                total_weekly_rev += gp_data_function.get_revenue(product)
                # total value of the product
            print(total_weekly_rev, ' total rev for product {}'.format(product))

            # NPV calculation
            total_wekly_rev = total_weekly_rev  / (1+self.weekly_interest_rate)**week_n
            # Records results
            total_rev_week.append( ('week = ',week_n, total_weekly_rev))
                # average value per customer
            rev_per_client_week.append( ('week = ',week_n,total_weekly_rev  / \
                     time_series[week_n][2]))

    def esp_open_money_market_bonus(self, client):
        """This is a simpy process for opening a money market bonus account.
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""

        # opening a money market bonus account

        open_mmb = self.esp_money_market_bonus_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_mmb
        client.have_mmb = 1
        client.esp_open_money_market_bonus_request = open_mmb


    def esp_open_collateral_mma(self, client):
        """This is a simpy process for opening a open collateral mma accounts
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""

        open_cmma = self.esp_collateral_mma_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_cmma
        client.have_cmma = 1
        client.esp_open_collateral_mma_request = open_cmma


    def esp_open_cash_management(self, client):
        """This is a simpy process for opening a cash management checking account
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""


        open_cmc = self.esp_cash_management_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_cmc
        client.have_cm = 1
        client.esp_open_cash_management_request = open_cmc


    def esp_open_fx(self, client):
        """This is a simpy process for opening a fx-account account
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""


        open_fx = self.esp_fx_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_fx
        client.have_fx = 1
        client.esp_open_fx_request = open_fx


    def esp_open_letters_of_credit(self, client):
        """This is a simpy process for opening a letters of credit
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""


        open_letter_credit = self.esp_letters_of_credit_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_letter_credit
        client.have_loc = 1
        client.esp_open_letters_of_credit_request = open_letter_credit


    def esp_open_enterprise_sweep(self, client):
        """This is a simpy process for opening a letters of credit
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""


        open_es = self.esp_enterprise_sweep_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_es
        client.have_es = 1
        client.esp_open_enterprise_sweep_request = open_es



    def esp_open_checking(self, client):
        """This is a simpy process for opening a letters of credit
        Also, append the resource request for simpy to the client object.
        This will let us release this resource request later"""

        open_checking = self.esp_checking_resource.request()
        # Wait until its our turn or until or the customer churns
        yield open_checking
        client.have_checking = 1
        client.esp_open_checking_request = open_checking

    def esp_close_checking(self, client):
        """This releases the resource request for the SImpy resource representing
        checking. """
        print('closing checking')
        client.have_checking = 0
        yield self.esp_checking_resource.release(client.esp_open_checking_request)

    def esp_close_cash_management(self, client):
        """This releases the resource request for the Simpy resource
        representing cash management."""
        print('closing cash management')
        client.have_cm = 0
        yield self.esp_cash_management_resource.release(client.esp_open_cash_management_request)

    def esp_close_collateral_mma(self, client):
        """This releases the resource request for the Simpy resource representing
        collateral mma accounts"""
        print('closing collateral mma')
        client.have_cmma = 0
        yield self.esp_collateral_mma_resource.release(client.esp_open_collateral_mma_request)

    def esp_close_enterprise_sweep(self, client):
        """This releases the resource request for the SImpy resource representing
        enterprise sweep"""
        print('closing enterprise sweep')
        client.have_es = 0
        yield self.esp_enterprise_sweep_resource.release(client.esp_open_enterprise_sweep_request)

    def esp_close_letters_of_credit(self, client):
        """This releases the resource request for the Simpy resource representign
        letters of credit"""
        print('closing letters of credit')
        client.have_loc = 0
        yield self.esp_letters_of_credit_resource.release(client.esp_open_letters_of_credit_request)

    def esp_close_money_market_bonus(self, client):
        """This releases teh resource request for the Simpy resource representing
        money market bonus accounts"""
        print('closing money market bonus')
        client.have_mmb = 0
        yield self.esp_money_market_bonus_resource.release(client.esp_open_money_market_bonus_request)

    def esp_close_fx(self, client):
        """This releases the resource request for the Simpy resource representing
        foreign exchange products"""
        print('closeing fx')
        client.have_fx = 0
        yield self.esp_fx_resource.release(client.esp_open_fx_request)



    def close_accounts(self, client):
        """Release the simpy process for each of the Simpy Products.
        This occurs once a client has churned.
        In addition, remove this client from the list of clients that we have"""
        yield self.env.timeout(client.client_lifetime)
        print()
        print('WE are closing accounts for client {}'.format(client.client_id))
        print(len(self.list_of_all_clients),' length of client list before')
        self.list_of_all_clients.remove(client)
        print(len(self.list_of_all_clients),'len list of all cleitns')
        # Drop the clients from each product once they churn
        yield self.esp_cash_management_resource.release(client.esp_open_cash_management_request)
        yield self.esp_checking_resource.release(client.esp_open_checking_request)
        yield self.esp_collateral_mma_resource.release(client.esp_open_collateral_mma_request)
        yield self.esp_enterprise_sweep_resource.release(client.esp_open_enterprise_sweep_request)
        yield self.esp_letters_of_credit_resource.release(client.esp_open_letters_of_credit_request)
        yield self.esp_money_market_bonus_resource.release(client.esp_open_money_market_bonus_request)
        yield self.esp_fx_resource.release(client.esp_open_fx_request)
        yield self.esp_client_alive_resource.release(
            client.esp_client_alive_resource_request)



if __name__ == "__main__":
    sys.stdout.flush()
    env = simpy.Environment()
    start = time.time()
    n_weeks_run = 104
    trials = 3
    federal_funds_rate = .0075 # May 11, 2017
    inflation_rate = .025 # March 2017
    # Evidence for staring products
    starting_evidence = {'cash_management':1,'checking_usd':1}
    #starting_evidence = None
    # keep a list of the data attributes over time
    times_series_all_clients = []
    times_series_cumulative_clients = []
    # products
    time_series_money_market_bonus = []
    time_series_esp_collateral_mma = []
    time_series_esp_cash_management = []
    time_series_esp_fx = []
    time_series_esp_letters_of_credit = []
    time_series_esp_enterprise_sweep = []
    time_series_esp_checking = []
    # GP
    time_series_esp_money_market_bonus_total_weekly_rev= []
    time_series_esp_money_market_bonus_rev_per_customer = []
    time_series_esp_collateral_mma_total_weekly_rev = []
    time_series_esp_collateral_mma_rev_per_customer = []
    time_series_esp_cash_management_total_weekly_rev = []
    time_series_esp_cash_management_rev_per_customer = []
    time_series_esp_fx_total_weekly_rev = []
    time_series_esp_fx_rev_per_customer = []
    time_series_esp_letters_of_credit_total_weekly_rev = []
    time_series_esp_letters_of_credit_rev_per_customer = []
    time_series_esp_enterprise_sweep_total_weekly_rev = []
    time_series_esp_enterprise_sweep_rev_per_customer = []
    time_series_esp_checking_total_weekly_rev = []
    time_series_esp_checking_rev_per_customer = []

    #for i in range(3):



    # Record data over multiple runs
    for i in range(trials):
        print()
        print('Starting simulation {}'.format(i))
        print()
        esp_flow = ESP_flow(env,
                        number_of_weeks_to_run = n_weeks_run,
                        yearly_interest_rate = federal_funds_rate * inflation_rate,
                        evidence_ = starting_evidence)




        env.process(esp_flow.initiate_week_client_run())
        env.run()
        # Keep track of the data for each run
        times_series_all_clients.append(esp_flow.time_series_total_clients)
        times_series_cumulative_clients.append(esp_flow.time_series_cumulative_clients)
        # products
        time_series_money_market_bonus.append(esp_flow .time_series_esp_money_market_bonus)
        time_series_esp_collateral_mma.append(esp_flow .time_series_esp_collateral_mma)
        time_series_esp_cash_management.append(esp_flow .time_series_esp_cash_management)
        time_series_esp_fx.append(esp_flow .time_series_esp_fx)
        time_series_esp_letters_of_credit.append(esp_flow .time_series_esp_letters_of_credit)
        time_series_esp_enterprise_sweep.append(esp_flow .time_series_esp_enterprise_sweep)
        time_series_esp_checking.append(esp_flow .time_series_esp_checking)
        # GP
        time_series_esp_money_market_bonus_total_weekly_rev.append(esp_flow .time_series_esp_money_market_bonus_total_weekly_rev)
        time_series_esp_money_market_bonus_rev_per_customer.append(esp_flow.time_series_esp_money_market_bonus_rev_per_customer)
        time_series_esp_collateral_mma_total_weekly_rev.append(esp_flow.time_series_esp_collateral_mma_total_weekly_rev)
        time_series_esp_collateral_mma_rev_per_customer.append(esp_flow.time_series_esp_collateral_mma_rev_per_customer)
        time_series_esp_cash_management_total_weekly_rev.append(esp_flow.time_series_esp_cash_management_total_weekly_rev)
        time_series_esp_cash_management_rev_per_customer.append(esp_flow.time_series_esp_cash_management_rev_per_customer)
        time_series_esp_fx_total_weekly_rev.append(esp_flow.time_series_esp_fx_total_weekly_rev)
        time_series_esp_fx_rev_per_customer.append(esp_flow.time_series_esp_fx_rev_per_customer)
        time_series_esp_letters_of_credit_total_weekly_rev.append(esp_flow.time_series_esp_letters_of_credit_total_weekly_rev)
        time_series_esp_letters_of_credit_rev_per_customer.append(esp_flow.time_series_esp_letters_of_credit_rev_per_customer)
        time_series_esp_enterprise_sweep_total_weekly_rev.append(esp_flow.time_series_esp_enterprise_sweep_total_weekly_rev)
        time_series_esp_enterprise_sweep_rev_per_customer.append(esp_flow.time_series_esp_enterprise_sweep_rev_per_customer)
        time_series_esp_checking_total_weekly_rev.append(esp_flow.time_series_esp_checking_total_weekly_rev)
        time_series_esp_checking_rev_per_customer.append(esp_flow.time_series_esp_checking_rev_per_customer)


        print()
        print("SUMMARY STATISTICS")
        print('Finished at time {}'.format(env.now))

        print('Time series of total clients over time = {}'.format(
            esp_flow.time_series_total_clients
        ))
        print('Time series of cumulative clients over time = {}'.format(
            esp_flow.time_series_cumulative_clients
        ))
        print("Time series of esp money market bonus {} ".format(
            esp_flow .time_series_esp_money_market_bonus))
        print("Time series of esp collateral mma {} ".format(
            esp_flow .time_series_esp_collateral_mma))
        print("Time series of esp cash management {} ".format(
            esp_flow .time_series_esp_cash_management))
        print("Time series of esp fx{} ".format(
                    esp_flow .time_series_esp_fx))
        print("Time series of esp letters of credit {} ".format(
                        esp_flow .time_series_esp_letters_of_credit))
        print("Time series of esp enterprise sweep {} ".format(
                    esp_flow .time_series_esp_enterprise_sweep))
        print("Time series of esp checking {} ".format(
                esp_flow .time_series_esp_checking))
        print("Total rgp for esp money market bonus per week {}".format(
        esp_flow .time_series_esp_money_market_bonus_total_weekly_rev))
        print("GP per custome rfor esp money market bonus per week {}".format(
            esp_flow.time_series_esp_money_market_bonus_rev_per_customer
        ))
        print()
        print("GP per total {} and per customer for collateral MMA  {}".format(
                esp_flow.time_series_esp_collateral_mma_total_weekly_rev,
                esp_flow.time_series_esp_collateral_mma_rev_per_customer
        ))
        print()
        print('GP for cash management total {} and gp cash management per customer {}'.format(
            esp_flow.time_series_esp_cash_management_total_weekly_rev,
            esp_flow.time_series_esp_cash_management_rev_per_customer
        ))
        print()
        print(' GP for fx total {} and fx per client {}'.format(
            esp_flow.time_series_esp_fx_total_weekly_rev,
            esp_flow.time_series_esp_fx_rev_per_customer))
        print()
        print('GP fox letters of credit toal {} and gp for letters of credit per customer {}'.format(
        esp_flow.time_series_esp_letters_of_credit_total_weekly_rev,
                    esp_flow.time_series_esp_letters_of_credit_rev_per_customer
        ))
        print()
        print('GP for enterprise sweep total {} and enterprise sweep gP per client{}'.format(
                esp_flow.time_series_esp_enterprise_sweep_total_weekly_rev,
                esp_flow.time_series_esp_enterprise_sweep_rev_per_customer
        ))
        print()
        print('GP for checking total {} and checking per client per week {} '.format(
                esp_flow.time_series_esp_checking_total_weekly_rev,
                esp_flow.time_series_esp_checking_rev_per_customer
        ))
        end = time.time()
        print('{} weeks tooks {} seconds'.format(n_weeks_run,end-start))


    # Save the generated data
    with open('data-evidence-checking-cm/time_series_all_clients', 'wb') as fp:
        pickle.dump(times_series_all_clients, fp)
    with open('data-evidence-checking-cm/times_series_cumulative_clients', 'wb') as fp:
        pickle.dump(times_series_cumulative_clients, fp)
    # products
    with open('data-evidence-checking-cm/time_series_money_market_bonus', 'wb') as fp:
        pickle.dump(time_series_money_market_bonus, fp)
    with open('data-evidence-checking-cm/time_series_esp_collateral_mma ', 'wb') as fp:
        pickle.dump(time_series_esp_collateral_mma, fp)
    with open('data-evidence-checking-cm/time_series_esp_cash_management', 'wb') as fp:
        pickle.dump(time_series_esp_cash_management , fp)
    with open('data-evidence-checking-cm/time_series_esp_fx', 'wb') as fp:
        pickle.dump(time_series_esp_fx , fp)
    with open('data-evidence-checking-cm/time_series_esp_letters_of_credit ', 'wb') as fp:
        pickle.dump(time_series_esp_letters_of_credit  , fp)
    with open('data-evidence-checking-cm/time_series_esp_enterprise_sweep', 'wb') as fp:
        pickle.dump(time_series_esp_enterprise_sweep  , fp)
    with open('data-evidence-checking-cm/time_series_esp_checking', 'wb') as fp:
        pickle.dump(time_series_esp_checking , fp)
    # GP
    with open('data-evidence-checking-cm/time_series_esp_money_market_bonus_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_money_market_bonus_total_weekly_rev, fp)
    with open('data-evidence-checking-cm/time_series_esp_money_market_bonus_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_money_market_bonus_rev_per_customer,fp)
    with open('data-evidence-checking-cm/time_series_esp_collateral_mma_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_collateral_mma_total_weekly_rev , fp)
    with open('data-evidence-checking-cm/time_series_esp_collateral_mma_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_collateral_mma_rev_per_customer , fp)
    with open('data-evidence-checking-cm/time_series_esp_cash_management_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_cash_management_total_weekly_rev  , fp)
    with open('data-evidence-checking-cm/time_series_esp_cash_management_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_cash_management_rev_per_customer   , fp)
    with open('data-evidence-checking-cm/time_series_esp_fx_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_fx_total_weekly_rev , fp)

    with open('data-evidence-checking-cm/time_series_esp_fx_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_fx_rev_per_customer, fp)
    with open('data-evidence-checking-cm/time_series_esp_letters_of_credit_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_letters_of_credit_total_weekly_rev, fp)
    with open('data-evidence-checking-cm/time_series_esp_letters_of_credit_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_letters_of_credit_rev_per_customer, fp)
    with open('data-evidence-checking-cm/time_series_esp_enterprise_sweep_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_enterprise_sweep_total_weekly_rev , fp)
    with open('data-evidence-checking-cm/time_series_esp_enterprise_sweep_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_enterprise_sweep_rev_per_customer  , fp)
    with open('data-evidence-checking-cm/time_series_esp_checking_total_weekly_rev', 'wb') as fp:
        pickle.dump(time_series_esp_checking_total_weekly_rev , fp)
    with open('data-evidence-checking-cm/time_series_esp_checking_rev_per_customer', 'wb') as fp:
        pickle.dump(time_series_esp_checking_rev_per_customer, fp)
