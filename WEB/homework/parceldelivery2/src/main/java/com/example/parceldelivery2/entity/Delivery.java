package com.example.parceldelivery2.entity;

import lombok.Data;
import lombok.NoArgsConstructor;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import java.sql.Date;

@Entity
@Data
@NoArgsConstructor
public class Delivery {

    public Delivery(String sender, String recipient,String payment,String storages,
                    String address, Integer entrancepassword,String phone,String email,Date dates,Date visit,
                    String tax,String receipt){
        this.sender = sender;
        this.recipient = recipient;
        this.payment = payment;
        this.storages = storages;
        this.address =address;
        this.entrancepassword = entrancepassword;
        this.phone = phone;
        this.email = email;
        this.dates =dates;
        this.visit = visit;
        this.tax = tax;
        this.receipt = receipt;
    }


    @Id //프라이머리키를 의미하는것이다.
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer id;

    private String sender;
    private String recipient;
    private String payment;
    private String storages;
    private String address;
    private Integer entrancepassword;
    private String phone;
    private String email;
    private Date dates;
    private Date visit;
    private String tax;
    private String receipt;

}
