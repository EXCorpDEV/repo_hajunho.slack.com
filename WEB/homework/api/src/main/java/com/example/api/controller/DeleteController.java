package com.example.api.controller;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/delete-api")
public class DeleteController {

    @DeleteMapping("/{variable}")
    public String DeleteVariable(@PathVariable String variable){
        return variable;
    }

    @DeleteMapping("/request1")
    public String getRequesParam1(@RequestParam String email){
        return  "email :" + email;
    }

}
