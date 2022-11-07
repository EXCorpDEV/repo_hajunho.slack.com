package com.example.parceldelivery2.entity;

import lombok.Data;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import java.sql.Date;

@Entity
@Data
public class Delivery {

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
