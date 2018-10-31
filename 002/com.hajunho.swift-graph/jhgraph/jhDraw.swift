//
//  jhDraw.swift
//  com.hajunho.swift-graph
//
//  Created by Junho HA on 2018. 10. 15..
//  Copyright © 2018년 hajunho.com. All rights reserved.
//

import UIKit

internal extension UIColor {
    
    convenience init(red: Int, green: Int, blue: Int) {
        assert(red >= 0 && red <= 255, "Invalid red component")
        assert(green >= 0 && green <= 255, "Invalid green component")
        assert(blue >= 0 && blue <= 255, "Invalid blue component")
        
        self.init(red: CGFloat(red) / 255.0, green: CGFloat(green) / 255.0, blue: CGFloat(blue) / 255.0, alpha: 1.0)
    }
    
    convenience init(rgb: Int) {
        self.init(
            red: (rgb >> 16) & 0xFF,
            green: (rgb >> 8) & 0xFF,
            blue: rgb & 0xFF
        )
    }
}

class jhDraw : UIView {
    
    internal static let maxR : CGFloat = 10000.0 //will be deleted
    internal static let ARQ : CGFloat = 10000.0 // standard value to calculate x, y position
    
    static var color : CGColor = UIColor.blue.cgColor
    
    struct _xy {
        var x : CGFloat
        var y : CGFloat
        init(_ x : CGFloat, _ y : CGFloat) {
            self.x = x
            self.y = y
        }
    }
    
    static func jhColor(r:CGFloat , g:CGFloat , b:CGFloat , a:Float) -> CGColor {
        return  UIColor(red: r / 255.0, green: g / 255.0, blue: b / 255.0, alpha: r).cgColor
    }
    
    static func worldLine(context : CGContext?, _ x1 : Int, _ y1 : Int, _ x2 : Int, _ y2 : Int, _ lineWidth : CGFloat) {
        context?.move(to: CGPoint(x: x1, y: y1))
        context?.addLine(to: CGPoint(x: x2, y: y2))
        context?.setStrokeColor(self.color)
        context?.setLineWidth(lineWidth)
        context?.strokePath()
    }
    
    static func worldLine(context : CGContext?, _ x1 : Int, _ y1 : Int, _ x2 : Int, _ y2 : Int, _ lineWidth : CGFloat, _ color : CGColor) {
        context?.move(to: CGPoint(x: x1, y: y1))
        context?.addLine(to: CGPoint(x: x2, y: y2))
        context?.setStrokeColor(color)
        context?.setLineWidth(lineWidth)
        context?.strokePath()
    }
    
    static func worldLine(context : CGContext?, _ x1 : CGFloat, _ y1 : CGFloat, _ x2 : CGFloat, _ y2 : CGFloat, _ lineWidth : CGFloat, _ color : CGColor) {
        if GS.shared.logLevel.contains(.graph) { print("draw_worldLine_\(x1), \(y1), \(x2), \(y2)")}
        context?.move(to: CGPoint(x: x1, y: y1))
        context?.addLine(to: CGPoint(x: x2, y: y2))
        context?.setStrokeColor(color)
        context?.setLineWidth(lineWidth)
        context?.strokePath()
    }
    
    static func worldEllipse(context : CGContext?, _ x : CGFloat, _ y : CGFloat, _ width : CGFloat, _ height : CGFloat, _ lineWidth : CGFloat, _ color : CGColor) {
        if GS.shared.logLevel.contains(.graph) { print("draw_worldEllipse_\(x), \(y), \(width), \(height)")}
        context?.addEllipse(in: CGRect(x: x - width/2 , y: y - height/2, width: width, height: height))
        context?.setStrokeColor(color)
        context?.setLineWidth(lineWidth)
        context?.strokePath()
    }
    
    static func setColor(color : CGColor) {
        self.color = color
    }
    
    func getX(_ x : CGFloat, panelWidth : CGFloat, fixedPanelWidth : CGFloat ) -> CGFloat? {
        var retX : CGFloat? = nil
        retX = x * panelWidth / fixedPanelWidth
        return retX
    }
    
    func getY(_ y : CGFloat, panelHeight : CGFloat, fixedPanelHeight : CGFloat ) -> CGFloat? {
        var retY : CGFloat? = nil
        retY = y * panelHeight / fixedPanelHeight
        return retY
    }
    
}

