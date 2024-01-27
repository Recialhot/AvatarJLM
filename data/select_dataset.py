#ysq
def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['amass_p1', 'amass_p2', 'amass_p3']:
        from data.dataset_amass import AMASS_Dataset as D
    elif dataset_type in ['tracking']:
        from data.dataset_tracking import TrackingData as D
    elif dataset_type in ['odt']:
        from data.dataset_odt import OdtData as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))
    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset

