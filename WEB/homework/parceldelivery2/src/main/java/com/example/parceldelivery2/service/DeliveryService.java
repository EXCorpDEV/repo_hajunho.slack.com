package com.example.parceldelivery2.service;

import com.example.parceldelivery2.entity.Delivery;
import com.example.parceldelivery2.repository.DeliveryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class DeliveryService {
    @Autowired
    private DeliveryRepository deliveryRepository;
    // 글작성
    public void write(Delivery delivery){
        deliveryRepository.save(delivery);
    }

    public List<Delivery> deliveryList(){
       return deliveryRepository.findAll();
    }

    public Delivery deliveryView(Integer id){
        return deliveryRepository.findById(id).get();
    }
}
