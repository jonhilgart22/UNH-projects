#! usr/bin/env python
import scipy.stats as stats

esp_eligible_products = ['Money Market Bonus','Collateral MMA',
'Cash Management',
'FX Products',
'Letters of Credit',
'Enterprise Sweep',
'Checking USD']



class ESP_revenue_predictions(object):
    """Generate weekly revenue predictions for a given esp product"""
    def __init__(self):
        pass
    @staticmethod
    def get_revenue(product):
        """Get the revenue of associated products"""
        if product == 'mmb':
            return ESP_revenue_predictions.money_market_bonus_weekly_rev()
        elif product == 'cmma':
            return ESP_revenue_predictions.collateral_mma_weekly_rev()
        elif product == 'cm':
            return ESP_revenue_predictions.cash_management_weekly_rev()
        elif product == 'fx':
            return ESP_revenue_predictions.fx_weekly_rev()
        elif product == 'loc':
            return ESP_revenue_predictions.letters_of_credit_weekly_rev()
        elif product == 'es':
            return ESP_revenue_predictions.enterprise_sweep_weekly_rev()
        elif product =='checking':
            return ESP_revenue_predictions.checking_weekly_rev()

    @staticmethod
    def money_market_bonus_weekly_rev(mean =1.5021545626038255 ,
        shape= 0.6456616031354403, scale_l= -45.02, scale_a=290.7614966933886):
        """ This gives a predicted weekly GP from an Exponential Weibull distribution.
        The default parameters here are from 2016."""
        mmb_weekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        return mmb_weekly

    @staticmethod
    def collateral_mma_weekly_rev(mean =12.199742430501765,
        shape= 0.31096239732167663, scale_l= -0.18002171203530209,
         scale_a=0.43735527761239656):
        """ This gives a predicted weekly GP from and Exponential Weibull distribution.
        The default parameters here are from 2016. Collaterall MMA can not be negative"""
        cmma_weekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        if cmma_weekly<0:
            cmma_weekly = 0
        return cmma_weekly

    @staticmethod
    def cash_management_weekly_rev(mean =57.972170036069599,
        shape=0.41151231738056693 ,scale_l= -137.92721250972454,
         scale_a=7.7367732606194721):
        """ This gives a predicted weekly GP from and Exponential Weibull distribution.
        The default parameters here are from 2016"""
        cm_mweekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        return cm_mweekly

    @staticmethod
    def fx_weekly_rev(mean =169.62798971892437,
        shape= 0.30030155042572004,scale_l=  -172.28912908419767,
         scale_a= 1.2426950697449026):
        """ This gives a predicted weekly GP from and Exponential Weibull distribution.
        The default parameters here are from 2016"""
        fx_mweekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        return fx_mweekly

    @staticmethod
    def letters_of_credit_weekly_rev(one = -663.9800000000298, two = 869.07882491183921):
        """ This gives a predicted weekly GP from and Exponential  distribution.
        The default parameters here are from 2016"""
        loc_mweekly = stats.expon.rvs(one,two)/4
        return loc_mweekly

    @staticmethod
    def enterprise_sweep_weekly_rev(mean =0.94859348495630935,
        shape=0.6405443331103936, scale_l= -5.1857408750582383e-30,
         scale_a= 160.9137385022949):
        """ This gives a predicted weekly GP from and Exponential Weibull distribution.
        The default parameters here are from 2016. Enterprise sweep can not have
        negative values"""
        es_weekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        if es_weekly <0:
            es_weekly=0
        return es_weekly

    @staticmethod
    def checking_weekly_rev(mean =1.131475921473887,
        shape=0.72178136928413106, scale_l= -34.990,
         scale_a= 185.81704752407666):
        """ This gives a predicted weekly GP from and Exponential Weibull distribution.
        The default parameters here are from 2016"""
        checking_weekly = stats.exponweib.rvs(mean,shape,scale_l,scale_a)/4
        return checking_weekly




if __name__ == '__main__':
    print('weekly ESP money market bonus GP' ,
          ESP_revenue_predictions.money_market_bonus_weekly_rev())
    print('weekly ESP collateral mma GP' ,
          ESP_revenue_predictions.collateral_mma_weekly_rev())
    print('weekly ESP cash management GP' ,
          ESP_revenue_predictions.cash_management_weekly_rev())
    print('weekly ESP fx GP' ,
          ESP_revenue_predictions.fx_weekly_rev())
    print('weekly ESP letters of credit GP' ,
              ESP_revenue_predictions.letters_of_credit_weekly_rev())
    print('weekly ESP enterprise sweep GP' ,
              ESP_revenue_predictions.enterprise_sweep_weekly_rev())
    print('weekly ESP checking GP' ,
              [ESP_revenue_predictions.checking_weekly_rev() for _ in range(40)],
              sum([ESP_revenue_predictions.checking_weekly_rev() for _ in range(40)]))
