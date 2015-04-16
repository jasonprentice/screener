#import xml.etree.ElementTree as ET
import lxml.etree as ET
import gzip
import os
import re
import time

def capture_property(text, tag, attrib_name='', attrib_value=''):
    if attrib_name:
        regexp = '<' + tag + ' [ .]*' + attrib_name + '=[\'\"]' + attrib_value + '[\'\"][^>]*>([^<]*)</' + tag + '>'        
    else:
        regexp = '<' + tag + ' [ ^>]*>([^<]*)</' + tag + '>'

    
    return re.findall(regexp, text)

def unpack(fname):
    if fname[-3:] == '.gz':
        return gzip.open(fname)
    else:
        return open(fname,'r')

def extract_namespace(filename):    
    xbrl_text = unpack(filename).read()
    ns_keys = re.findall('xmlns:([^=]*)=', xbrl_text)
    ns = {}
    rev_ns = {}    
    for key in ns_keys:        
        resolved = re.search('xmlns:' + key + '=[\'"]([^[\'"]*)[\'"]', xbrl_text).group(1)
        ns[key] = resolved
        rev_ns[resolved] = key

    return (ns, rev_ns)

def rule2str(rule):
    s = ''
    for tag, wt in rule:        
        if wt == 1.0:
            s = s + ' + ' + tag
        elif wt == -1.0:
            s = s + ' - ' + tag
        else:
            s = s + ' ' + str(wt) + ' * ' + tag
    return s

class parseXBRL:
    def __init__(self):
        self.d_context = None
        self.i_context = None
        self.data_dir = '../data/edgar/'
        gaap_dir = '../data/us-gaap-2014-01-31/stm/'
        gaap_calfiles = ['us-gaap-stm-sfp-cls-cal-2014-01-31.xml',
                         'us-gaap-stm-soi-cal-2014-01-31.xml',
                         'us-gaap-stm-scf-dir-cal-2014-01-31.xml']
                         #'us-gaap-stm-scf-indir-cal-2014-01-31.xml']


        self.scf_items = ['NetCashProvidedByUsedInOperatingActivities',
                          'NetCashProvidedByUsedInFinancingActivities',
                          'NetCashProvidedByUsedInInvestingActivities',                  
                          'EffectOfExchangeRateOnCashAndCashEquivalents',
                          'CashAndCashEquivalentsPeriodIncreaseDecrease']

        self.soi_items = ['Revenues',
                          'CostOfRevenue', 
                          'GrossProfit',
                          'OperatingExpenses',
                          'OtherOperatingIncome',
                          'OperatingIncomeLoss',
                          'NonoperatingIncomeExpense',
                          'InterestAndDebtExpense',
                          'IncomeLossFromEquityMethodInvestments',
                          'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
                          'IncomeTaxExpenseBenefit',
                          'IncomeLossFromContinuingOperationsIncludingPortionAttributableToNoncontrollingInterest',
                          'IncomeLossFromDiscontinuedOperationsNetOfTax',
                          'ExtraordinaryItemNetOfTax',
                          'ProfitLoss',
                          'NetIncomeLossAttributableToNoncontrollingInterest',
                          'NetIncomeLoss',
                          'PreferredStockDividendsAndOtherAdjustments',
                          'NetIncomeLossAvailableToCommonStockholdersBasic'
                          'CostsAndExpenses']                                               
                          


        self.sfp_items = ['AssetsCurrent',
                          'CashCashEquivalentsAndShortTermInvestments',
                          'ReceivablesNetCurrent',
                          'InventoryNet',
                          'PrepaidExpenseAndOtherAssetsCurrent',
                          'DeferredCostsCurrent',
                          'DerivativeInstrumentsAndHedges',
                          'AssetsNoncurrent',
                          'PropertyPlantAndEquipmentNet',
                          'Assets',
                          'LiabilitiesCurrent',
                          'AccountsPayableAndAccruedLiabilitiesCurrent',
                          'DebtCurrent',
                          'DerivativeInstrumentsAndHedgesLiabilities',
                          'LiabilitiesNoncurrent',
                          'LongTermDebtAndCapitalLeaseObligations',
                          'LiabilitiesOtherThanLongtermDebtNoncurrent',
                          'Liabilities',            
                          'CommitmentsAndContingencies',
                          'Equity',
                          'LiabilitiesAndStockholdersEquity',
                          'TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests',
                          'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest']
        self.all_items = self.scf_items + self.soi_items + self.sfp_items
                                                   
        self.gaap_rules = {}
    
        # Load rules defined in us-gaap cal files
        for filename in gaap_calfiles:
            self.gaap_rules = self.parseCalFile(gaap_dir + filename, self.gaap_rules)
        self.gaap_rules['us-gaap:Equity'] = [[('us-gaap:TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterests', 1.0), 
                                      ('us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 1.0)]]

        
        # self.gaap_rules = self.parseCalFile(gaap_dir + gaap_soi, self.gaap_rules)
        # self.gaap_rules = self.parseCalFile(gaap_dir + gaap_scf, self.gaap_rules)
        
        # for target,ruleset in self.gaap_rules.iteritems():
        #     for rule in ruleset:
        #         print target + ': ' + rule2str(rule)
    
        

    def parseCalFile(self, fname, all_rules):

        
        (ns, rev_ns) = extract_namespace(fname)        
        
        tree = ET.parse(unpack(fname))
        root = tree.getroot()
        for cal_link in root.iterfind('{*}calculationLink', namespaces=ns):
            tag_lookup = {}
            rules = {}
            for locator in cal_link.iterfind('{*}loc', namespaces=ns):
                tagname = locator.attrib['{'+ns['xlink']+'}href']
                tagname = tagname[tagname.find('#')+1:]
                tagname = tagname.replace('_',':')
                label = locator.attrib['{'+ns['xlink']+'}label']
                tag_lookup[label] = tagname
            for arc in cal_link.iterfind('{*}calculationArc',namespaces=ns):
                from_tag = tag_lookup[arc.attrib['{'+ns['xlink']+'}from']]
                to_tag = tag_lookup[arc.attrib['{'+ns['xlink']+'}to']]
                weight = float(arc.attrib['weight'])
                
                if from_tag not in rules.keys():
                    rules[from_tag] = []
                rules[from_tag].append( (to_tag, weight) )
        for tag, rule in rules.iteritems():
            if tag not in all_rules.keys():
                all_rules[tag] = [rule]
            else:
                all_rules[tag].append(rule)

        return all_rules

        
    def get_notes(self, cik, year, mo = None, report_type = '10-K'):
        self.load_xbrl(cik,year,mo,report_type)

        def strip_HTML(text):
            stripped_text = re.sub('&lt;','<',text)
            stripped_text = re.sub('&gt;','>',stripped_text) 
            stripped_text = re.sub('<[^>]*>',' ',stripped_text)
            stripped_text = re.sub('&[^;]*;',' ',stripped_text)
            stripped_text = stripped_text.replace('\n',' ')
            stripped_text = stripped_text.replace('\r',' ')
            return stripped_text

        textblocks = []
        for block in re.findall('<us-gaap:([^\s]+)TextBlock', self.xbrl_text):
            tagname = 'us-gaap:'+block+'TextBlock'            
            block_text = re.search('<'+tagname+' [^>]*>(.*)</'+tagname+'>', self.xbrl_text, flags=re.DOTALL).group(1)            
            block_text = strip_HTML(block_text)            
            wc = len(block_text.split())            
            textblocks.append( (block, block_text, wc) )
        
        return textblocks

    def load_xbrl(self, cik, year, mo=None, report_type='10-K'):
        directory = self.data_dir + str(cik) + '/' + str(year) + '/'
        if not mo:
            mo = [m for m in os.listdir(directory) if m[0] != '.']
            mo = mo[0]
        self.reportfile = directory + mo + '/' + report_type + '.xml.gz'
        self.calfile = directory + mo + '/' + report_type + '-cal.xml.gz'
        (self.ns, self.rev_ns) = extract_namespace(self.reportfile)
        self.xbrl_text = ' '.join(gzip.open(self.reportfile).read().split())
        xbrl = gzip.open(self.reportfile)
        self.root = ET.parse(xbrl).getroot()        
        
    def read_statement(self, cik, year, mo = None, report_type = '10-K'):        
        self.load_xbrl(cik,year,mo,report_type)
        directory = self.data_dir + str(cik) + '/' + str(year) + '/'
        # Infer contexts        
        self.get_contexts(year)        

        def sub_namespace(tag):            
            resolved = re.match('{([^}]*)}', tag).group(1)            
            return tag.replace('{' + resolved + '}', self.rev_ns[resolved] + ':')

        # Build dict mapping all items listed in financial statement to their values        
        def parse_tag(tag):
            key = sub_namespace(tag.tag)            
            value = tag.text      
            if value:      
                value = float(value)
            else:
                value = 0.0
            return (key, value)

        d_tags = self.root.findall('.//*[@contextRef="' + self.d_context + '"][@unitRef]')
        i_tags = self.root.findall('.//*[@contextRef="' + self.i_context + '"][@unitRef]')      
        statement_items = dict(map(parse_tag, d_tags + i_tags))
   
        # Load company-specific calculation rules
        self.co_rules = self.parseCalFile(directory + mo + '/' + report_type + '-cal.xml.gz', {})

        #@memoize
        def lookup(item):            
            def apply_rule(ruleset):
                total = 0.0
                for r in ruleset[0]:
                    total = total + r[1] * lookup(r[0])
                return total

            value = 0.0
            if item in statement_items.keys():
                value = statement_items[item]
            elif item in self.co_rules.keys():
                value = apply_rule(self.co_rules[item])
            elif item in self.gaap_rules.keys():
                value = apply_rule(self.gaap_rules[item])
                        
            return value

        item_vals = {}
        for item in self.sfp_items + self.soi_items + self.scf_items:
            item_vals[item] = lookup('us-gaap:' + item)

        try:
          num_shares = re.findall('<dei:EntityCommonStockSharesOutstanding [^>]*>([^<]*)</dei:EntityCommonStockSharesOutstanding>', self.xbrl_text)[0]
        except:
          num_shares = 0
        #num_shares = self.root.find('.//dei:EntityCommonStockSharesOutstanding', namespaces=self.ns).text
        item_vals['EntityCommonStockSharesOutstanding'] = int(float(num_shares))

        return item_vals

    

    def get_contexts(self,year):
        # Identify instant and duration context names for year        

        def find_contexts(items):                                    
            from collections import Counter
            contexts = Counter()
            for item in items:                                 
                rx1 = '<us-gaap:' + item + ' [^>]* contextRef=[\'\"]([^\'\"]+)[\'\"]'  # If other attributes are between tag and contextRef
                rx2 = '<us-gaap:' + item + ' contextRef=[\'\"]([^\'\"]+)[\'\"]'         # If contextRef immediately follows tag                
                
                tags = re.findall(rx1, self.xbrl_text)                                                     
                              
                if not tags:
                    tags = re.findall(rx2, self.xbrl_text)                                                
                
                for tag in tags:                      
                    contexts[tag] += 1
            

            if not contexts.values():              
              return None
            else:
              max_cnt = max(contexts.values())
              context_list = [context for (context,val) in contexts.iteritems() if val==max_cnt]                        
              
              return context_list
        

        def resolve_year(datetag, context_list):
            # Disambiguate year from a list of contexts
            tdict = {}            
            
            for context in context_list:                  
                incl_ns = True 
                if 'xbrli' in self.ns:
                  tag = self.root.find('.//xbrli:context[@id=\"' + context + '\"]', namespaces=self.ns)    
                  context_date = tag.find('.//xbrli:'+datetag, namespaces=self.ns).text.strip()
                else:                                    
                  tag = self.root.find('.//{*}context[@id=\"' + context + '\"]', namespaces=self.ns)    
                  
                  context_date = tag.find('.//{*}'+datetag, namespaces=self.ns).text.strip()
                  
                tdict[time.strptime(context_date,'%Y-%m-%d')] = context

            return tdict[max(tdict.keys())]

            #     split_date = context_date.split('-')
            #     context_year = split_date[0]
            #     context_mo = split_date[1]
            #     context_day = split_date[2]                                    
            #     if context_mo == '12' and context_day == '31':
            #         if int(context_year) == int(year)-1:
            #             return context                        
            #     else:
            #         if int(context_year) == int(year):
            #             return context
            # return None

        d_contexts = find_contexts(self.soi_items + self.scf_items) # Get duration context from income statement items
        if d_contexts is not None:
          self.d_context = resolve_year('endDate',d_contexts)       
        else:
          self.d_context = None

        i_contexts = find_contexts(self.sfp_items) # Get instant context from balance sheet items                
        if i_contexts is not None:
          self.i_context = resolve_year('instant',i_contexts)
        else:
          self.i_context = None
        




