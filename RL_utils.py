import torch

def calculate_grad_dot(grads, num_batchs):
    mgm = 0.0
    length = num_batchs // 2
    for name in grads:
        grad1 = torch.stack(grads[name][:length]).flatten()
        grad2 = torch.stack(grads[name][length: 2 * length]).flatten()
        grad1 -= grad1.mean()
        grad2 -= grad2.mean()
        
        mgm += torch.dot(grad1, grad2) / (num_batchs * length)
        
    return mgm
        
        
        
def MGM(model, criterion, optimizer, dataloader, num_batchs=50, num_params=50):
    from collections import defaultdict
    agg_mgm = 0.0
    grads = defaultdict(list)
    
    for i, (inputs, targets) in enumerate(dataloader):
        batch_size = targets.shape[0]
        
        if i >= num_batchs:
            break
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
    
        for name, param in model.named_parameters():
            if 'bias' in name:
                continue 
            grads[name].append(param.grad.view(-1)[:num_params].detach().clone())
            
    # divide grad vector into two 1D subvector with same length
    agg_mgm = calculate_grad_dot(grads, num_batchs)
    return agg_mgm

def MGM_SNN(model, criterion, optimizer, dataloader, num_batchs=50, num_params=50):
    from collections import defaultdict
    agg_mgm = 0.0
    grads = defaultdict(list)
    
    for i, (inputs, targets) in enumerate(dataloader):
        batch_size = targets.shape[0]
        
        if i >= num_batchs:
            break
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
    
        for name, param in model.named_parameters():
            if 'weight_v' not in name:
                continue 
            #b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')
            grads[name].append(param.grad.view(-1)[:num_params].detach().clone())
            
    # divide grad vector into two 1D subvector with same length
    agg_mgm = calculate_grad_dot(grads, num_batchs)
    return agg_mgm
    
            
            
            
            
        
        
        
        
    
    