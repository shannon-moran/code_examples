'''
Appendix B2: data_processing.py
For production code, we'd want to document in/output types, add unit tests, etc.
Not going to worry about that here.
'''

import pandas as pd
import numpy as np

def check_overlap(checkin,checkout,start_date,end_date):
    labels = []
    for i in checkin.index:
        if (end_date>checkin[i]>start_date): labels.append(True)
        elif (end_date>checkout[i]>start_date): labels.append(True)
        elif (checkin[i]<start_date) and (checkout[i]>end_date): labels.append(True)
        else: labels.append(False)
    assert(len(labels)==len(checkin))
    return labels

def label_2016_holidays(data):
    start = pd.to_datetime('2016-02-04')
    end = pd.to_datetime('2016-02-10')
    carnival = check_overlap(data.ds_checkin_first,data.ds_checkout_first,start,end)

    start = pd.to_datetime('2016-08-02') # preliminary events in some sports began 8/3
    end = pd.to_datetime('2016-08-23')
    olympics = check_overlap(data.ds_checkin_first,data.ds_checkout_first,start,end)

    data.loc[carnival,'holiday'] = 'carnival'
    data.loc[olympics,'holiday'] = 'olympics'
    data.holiday = data.holiday.fillna('normal');
    return data

def limit_to_2016(data):
    return data[data.ds_checkin_first<'12-01-2016']

def label_contact_outcomes(data):
    host_ignores = (data.ts_reply_at_first.isnull() & data.ts_accepted_at_first.isnull())
    host_denies = (data.ts_reply_at_first.notnull() & data.ts_accepted_at_first.isnull())
    host_accepts = (data.ts_reply_at_first.notnull() & data.ts_accepted_at_first.notnull())

    data.loc[host_ignores,'host_response'] = 'host_ignores'
    data.loc[host_denies,'host_response'] = 'host_denies'
    data.loc[host_accepts,'host_response'] = 'host_accepts'

    guest_ignores = (data.ts_accepted_at_first.notnull() & data.ts_booking_at.isnull())
    guest_books = (data.ts_accepted_at_first.notnull() & data.ts_booking_at.notnull())
    guest_denied = (data.ts_reply_at_first.notnull() & data.ts_accepted_at_first.isnull())
    guest_ignored = (data.ts_reply_at_first.isnull() & data.ts_accepted_at_first.isnull()) # same as host_ignores

    data.loc[guest_books,'guest_response'] = 'guest_books'
    data.loc[guest_ignores,'guest_response'] = 'guest_ignores'
    data.loc[guest_denied,'guest_response'] = 'guest_denied'
    data.loc[guest_ignored,'guest_response'] = 'guest_ignored'

    # Add a convert/not flag to the dataset.
    converted = (data.guest_response=='guest_books')
    data.loc[converted,'converted'] = 1

    not_converted = data.converted.isnull()
    data.loc[not_converted,'converted'] = 0

    return data

def label_listing_booking_types(data):
    # This is not the ideal way to do this-- but it does the trick
    c = 0
    for list_id in data.id_listing_anon.unique():
        c+=1
        if c%500==0: print(c)
        contact_types = data[data.id_listing_anon==list_id].contact_channel_first.unique()
        if ('instant_book' in contact_types) and ('book_it' in contact_types):
            data.loc[data.id_listing_anon==list_id,'listing_booking_type_diff'] = 1
        if 'instant_book' in contact_types:
            data.loc[data.id_listing_anon==list_id,'listing_booking_type'] = 'instant_yes'
        elif 'book_it' in contact_types:
            data.loc[data.id_listing_anon==list_id,'listing_booking_type'] = 'instant_no'
        else:
            data.loc[data.id_listing_anon==list_id,'listing_booking_type'] = 'contact_me'
    return data

def label_host_booking_types(data):
    # This is not the ideal way to do this-- but it does the trick
    c = 0
    for host_id in data.id_host_anon.unique():
        c+=1
        if c%500==0: print(c)
        contact_types = data[data.id_host_anon==host_id].listing_booking_type.unique()
        if ('instant_yes' in contact_types) and ('instant_no' in contact_types):
            data.loc[data.id_host_anon==host_id,'host_booking_type_diff'] = 1
        if 'instant_yes' in contact_types:
            data.loc[data.id_host_anon==host_id,'host_booking_type'] = 'instant_yes'
        elif 'instant_no' in contact_types:
            data.loc[data.id_host_anon==host_id,'host_booking_type'] = 'instant_no'
        else:
            data.loc[data.id_host_anon==host_id,'host_booking_type'] = '-unknown-'
    return data

def label_host_num_listings(data):
    # get number of listings per host
    grouped = data[['id_listing_anon','id_host_anon']].groupby('id_host_anon').nunique()
    host_num_listings = grouped['id_listing_anon'].reset_index().rename(columns={'id_listing_anon':'host_num_listings'})
    data = pd.merge(data,host_num_listings,how='left',on='id_host_anon')
    return data

def process_data(data):
    data = label_contact_outcomes(data)
    data = label_2016_holidays(data)
    data = limit_to_2016(data)
    print('starting listings')
    data = label_listing_booking_types(data)
    print('starting hosts')
    data = label_host_booking_types(data)
    print('almost done')
    data = label_host_num_listings(data)
    return data
