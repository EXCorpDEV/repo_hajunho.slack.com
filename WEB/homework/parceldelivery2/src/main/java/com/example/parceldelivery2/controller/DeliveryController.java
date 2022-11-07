package com.example.parceldelivery2.controller;

import com.example.parceldelivery2.entity.Delivery;
//import com.example.parceldelivery2.service.DeliveryService;
import com.example.parceldelivery2.service.DeliveryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;



@Controller
public class DeliveryController {
    @Autowired
    private DeliveryService deliveryService;

    @GetMapping("/delivery/write")
    public String deliveryWriteForm(){
        return "deliverywrite";
    }

    @PostMapping("/delivery/writepro")
    public String deliveryWritePro(Delivery delivery){
        deliveryService.write(delivery);
        return "";
    }

    @GetMapping("/delivery/list")
    public String deliveryList(Model model){
        model.addAttribute("list",deliveryService.deliveryList());
        return "deliverylist";
    }

    @GetMapping("delivery/view")
    public String deliveryView(Model model ,Integer id){
        model.addAttribute("delivery", deliveryService.deliveryView(id));
        return "deliveryview";
    }
}
